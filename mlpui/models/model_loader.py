import torch
import logging
from scripts.config.config_loader import ConfigBase
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.base import ContainerMetadata,Metadata
from omegaconf.nodes import AnyNode
from typing import Any
from collections import defaultdict
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

class ModelLoadingConfig(ConfigBase):
    yaml_file = "model_loader.yaml"
    safe_globals: list = []
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
    from mlpui.models.uma.inference import MLIPInferenceCheckpoint
    MLIPInferenceCheckpoint.__module__ = "fairchem.core.units.mlip_unit.api.inference"

    torch.serialization.add_safe_globals([ModelCheckpoint, scalar, dtype, Float64DType, encode,Metadata,MLIPInferenceCheckpoint,ListConfig,ContainerMetadata,AnyNode,Any,list,defaultdict,dict,int,DictConfig])
    logging.info("Checkpoint files will always be loaded safely.")

def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False,job="inference"):
    if device is None:
        device = torch.device("cpu")
    metadata = None

    torch_args = {}
    if MMAP_TORCH_FILES:
        torch_args["mmap"] = True

    pl_sd = torch.load(ckpt, map_location=device, weights_only=True, **torch_args)
    if hasattr(pl_sd, "ema_state_dict"):
        if job == "inference":
            state_dict = pl_sd.ema_state_dict
        elif job == "train":
            state_dict = pl_sd.model_state_dict
        elif job == "both":
            state_dict = {
                "ema": pl_sd.ema_state_dict,
                "train": pl_sd.model_state_dict,
            }
        metadata = {
            "model_config": pl_sd.model_config,
            "tasks_config": pl_sd.tasks_config,
        }
    elif isinstance(pl_sd, dict):
        if "state_dict" in pl_sd:
            state_dict = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                state_dict = pl_sd[key]
                if not isinstance(state_dict, dict):
                    state_dict = pl_sd
            else:
                state_dict = pl_sd
    else:
        state_dict = pl_sd
    return (state_dict, metadata) if return_metadata else state_dict

