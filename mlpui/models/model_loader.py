import torch
import logging

_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "C64": torch.complex64,
    "U64": torch.uint64,
    "U32": torch.uint32,
    "U16": torch.uint16,
}
from scripts.config.config_loader import ConfigBase

class ModelLoadingConfig(ConfigBase):
    yaml_file = "model_loading.yaml"
_cfg = ModelLoadingConfig()
MMAP_TORCH_FILES = _cfg.mmap_torch_files
DISABLE_MMAP = _cfg.disable_mmap



if True:  # ckpt/pt file whitelist for safe loading of old sd files
    class ModelCheckpoint:
        pass
    ModelCheckpoint.__module__ = "pytorch_lightning.callbacks.model_checkpoint"

    def scalar(*args, **kwargs):
        return None
    scalar.__module__ = "numpy.core.multiarray"

    from numpy import dtype
    from numpy.dtypes import Float64DType

    def encode(*args, **kwargs):  # no longer necessary on newer torch
        return None
    encode.__module__ = "_codecs"

    torch.serialization.add_safe_globals([ModelCheckpoint, scalar, dtype, Float64DType, encode])
    logging.info("Checkpoint files will always be loaded safely.")
