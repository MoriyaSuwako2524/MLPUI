import torch
from enum import Enum,auto
from scripts.config.config_loader import ConfigBase
import psutil
import logging

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
DISABLE_SMART_MEMORY = _cfg.disable_smart_memory
aimdo_enabled = _cfg.aimdo_enabled
if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")



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


def unet_inital_load_device(parameters, dtype):
    cpu_dev = torch.device("cpu")
    if aimdo_enabled:
        return cpu_dev

    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.SHARED:
        return torch_dev

    if DISABLE_SMART_MEMORY or vram_state == VRAMState.NO_VRAM:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024 #TODO
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_xpu + mem_free_torch
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_npu, _ = torch.npu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_npu + mem_free_torch
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_mlu, _ = torch.mlu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_mlu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total

def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except: #Old pytorch doesn't have .itemsize
            pass
    return dtype_size


def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")

def archive_model_dtypes(model):
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            setattr(module, f"{param_name}_comfy_model_dtype", param.dtype)
        for buf_name, buf in module.named_buffers(recurse=False):
            setattr(module, f"{buf_name}_comfy_model_dtype", buf.dtype)
