import os
import torch

CHECKPOINTS_DIR = os.path.expanduser("~/sam3/checkpoints")
BPE_PATH = os.path.expanduser("~/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")

CHECKPOINTS = {
    "sam1": os.path.join(CHECKPOINTS_DIR, "sam_vit_b_01ec64.pth"),
    "sam2": os.path.join(CHECKPOINTS_DIR, "sam2_hiera_base_plus.pt"),
    "sam3": os.path.join(CHECKPOINTS_DIR, "sam3.1_multiplex.pt"),
}

GPU = "gpu"
CPU = "cpu_only"
UNAVAILABLE = "unavailable"


def _detect_status() -> dict:
    cuda = torch.cuda.is_available()
    status = {}

    for name, ckpt in CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            status[name] = UNAVAILABLE
            continue
        try:
            if name == "sam1":
                import segment_anything  # noqa
            elif name == "sam2":
                import sam2  # noqa
            elif name == "sam3":
                import sam3  # noqa
            status[name] = GPU if cuda else CPU
        except ImportError:
            status[name] = UNAVAILABLE

    return status


class ModelRegistry:
    def __init__(self):
        self._status = _detect_status()
        self._instances: dict = {}

    def status(self) -> dict:
        return dict(self._status)

    def is_available(self, name: str) -> bool:
        return self._status.get(name, UNAVAILABLE) != UNAVAILABLE

    def get(self, name: str):
        if not self.is_available(name):
            raise ValueError(f"{name} is {self._status.get(name, UNAVAILABLE)}")
        if name not in self._instances:
            self._instances[name] = self._load(name)
        return self._instances[name]

    def _load(self, name: str):
        if name == "sam1":
            from app.models.sam1_model import SAM1Model
            return SAM1Model(CHECKPOINTS["sam1"])
        if name == "sam2":
            from app.models.sam2_model import SAM2Model
            return SAM2Model(CHECKPOINTS["sam2"])
        if name == "sam3":
            from app.models.sam3_model import SAM3Model
            return SAM3Model(CHECKPOINTS["sam3"], BPE_PATH)
        raise ValueError(f"Unknown model: {name}")


# module-level singleton
registry = ModelRegistry()
