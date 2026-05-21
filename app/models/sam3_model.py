import contextlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


from contextlib import contextmanager

# ZeroGPU's virtual GPU doesn't support CUDA memory pinning (cudaHostRegister).
# pin_memory() is a performance-only hint — safe to make it a no-op on failure.
_orig_pin_memory = torch.Tensor.pin_memory
def _safe_pin_memory(self, device=None):
    try:
        return _orig_pin_memory(self) if device is None else _orig_pin_memory(self, device)
    except RuntimeError:
        return self
torch.Tensor.pin_memory = _safe_pin_memory


@contextmanager
def _redirect_cuda_to_cpu():
    """
    Context manager that intercepts torch tensor-creation ops and redirects
    any hardcoded device='cuda' to 'cpu'.  SAM3's model_builder hardcodes
    device='cuda' in PositionEmbeddingSine, TransformerDecoder, and potentially
    other modules — this single context covers them all.
    """
    _ops = ["zeros", "ones", "empty", "arange", "linspace", "full", "rand", "randn", "tensor"]
    originals = {name: getattr(torch, name) for name in _ops}

    def _make_safe(orig_fn):
        def _safe(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return orig_fn(*args, **kwargs)
        return _safe

    for name, orig in originals.items():
        setattr(torch, name, _make_safe(orig))
    try:
        yield
    finally:
        for name, orig in originals.items():
            setattr(torch, name, orig)


def _patch_vitdet_for_float32():
    """
    Replace vitdet's BFloat16 fused MLP kernel with a float32 fallback.

    SAM3's addmm_act always casts inputs to BFloat16 for a fused GELU/ReLU kernel.
    Sam3Processor uses @inference_mode but NOT autocast, so float32 fc2 weights
    receive BFloat16 input → dtype mismatch on Turing and older GPUs.

    On Ampere+ we solve this by wrapping inference in torch.autocast(bfloat16),
    which auto-promotes float32 weights on the fly. On Turing (no native BF16
    compute, is_bf16_supported(including_emulation=False) == False) we instead
    patch vitdet's module namespace so Mlp.forward uses plain float32 ops.
    The patch is process-global but harmless: we only ever load one SAM3 instance.
    """
    import sam3.model.vitdet as vitdet_module

    def _addmm_act_float32(activation, linear, mat1):
        x = F.linear(mat1, linear.weight, linear.bias)
        if activation in (F.relu, torch.nn.ReLU):
            return torch.relu(x)
        if activation in (F.gelu, torch.nn.GELU):
            return F.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")

    vitdet_module.addmm_act = _addmm_act_float32


class SAM3Model:
    def __init__(self, checkpoint_path: str, bpe_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with _redirect_cuda_to_cpu():
            model = build_sam3_image_model(
                bpe_path=bpe_path,
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                device=self.device,
            )
            # Always cast weights to float32 and patch the fused MLP kernel.
            # On Ampere/H200, segment_with_text wraps inference in autocast(bfloat16)
            # which promotes ops on the fly; on Turing it stays float32 throughout.
            model = model.float()
            _patch_vitdet_for_float32()
            # Pass device explicitly so Sam3Processor's find_stage tensors land on CPU.
            self.processor = Sam3Processor(model, device=self.device, confidence_threshold=0.3)

    def segment_with_text(self, image: np.ndarray, prompt: str) -> list[dict]:
        """
        Run SAM3 native text grounding on image.
        Returns list of dicts with segmentation (bool HxW), bbox (xywh), area, score.
        """
        pil_image = PILImage.fromarray(image)
        # Defer bf16 check to call time: works on both local Turing (float32)
        # and ZeroGPU H200 (native bfloat16 inside @spaces.GPU context).
        bf16 = (
            self.device == "cuda"
            and torch.cuda.is_bf16_supported(including_emulation=False)
        )
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if bf16
            else contextlib.nullcontext()
        )
        with ctx:
            state = self.processor.set_image(pil_image)
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(state=state, prompt=prompt)

        masks = state.get("masks")      # tensor [N, 1, H, W] bool
        boxes = state.get("boxes")      # tensor [N, 4] x0y0x1y1
        scores = state.get("scores")    # tensor [N]

        if masks is None or len(masks) == 0:
            return []

        results = []
        for i in range(len(masks)):
            seg = masks[i, 0].cpu().numpy().astype(bool)  # H×W
            x0, y0, x1, y1 = boxes[i].cpu().tolist()
            bbox_xywh = [x0, y0, x1 - x0, y1 - y0]
            results.append({
                "segmentation": seg,
                "area": int(seg.sum()),
                "bbox": bbox_xywh,
                "score": float(scores[i].cpu()),
            })

        results.sort(key=lambda m: m["score"], reverse=True)
        return results

    def move_to(self, device: str):
        self.processor.model.to(device)
        self.processor.device = device
        self.device = device
        # find_stage holds template tensors (img_ids, text_ids) that must be on
        # the same device as the model; move them explicitly.
        fs = self.processor.find_stage
        if fs.img_ids is not None:
            fs.img_ids = fs.img_ids.to(device)
        if fs.text_ids is not None:
            fs.text_ids = fs.text_ids.to(device)
