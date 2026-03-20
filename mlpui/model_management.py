import torch
from enum import Enum,auto
from scripts.config.config_loader import ConfigBase
class VRAMState(Enum):
    DISABLED = 0    #No vram present: no need to move models to vram
    NO_VRAM = 1     #Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      #No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM


class ModelManagementConfig(ConfigBase):
    yaml_file = "model_management.yaml"

_cfg = ModelManagementConfig()
directml_enabled = _cfg.directml_enabled
xpu_available = _cfg.xpu_available
npu_available = _cfg.npu_available
mlu_available = _cfg.mlu_available
cpu = _cfg.cpu

if cpu:
    cpu_state = CPUState.CPU
else:
    cpu_state = CPUState.GPU



def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False

def is_ascend_npu():
    global npu_available
    if npu_available:
        return True
    return False

def is_mlu():
    global mlu_available
    if mlu_available:
        return True
    return False


def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        elif is_ascend_npu():
            return torch.device("npu", torch.npu.current_device())
        elif is_mlu():
            return torch.device("mlu", torch.mlu.current_device())
        else:
            return torch.device(torch.cuda.current_device())



def unet_dtype(device=None, model_params=0, supported_dtypes=[torch.float32], weight_dtype=None):
    if model_params < 0:
        model_params = 1000000000000000000000

    return torch.float32


def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[torch.float32]):
    if weight_dtype == torch.float32 or weight_dtype == torch.float64:
        return None


    return torch.float32
