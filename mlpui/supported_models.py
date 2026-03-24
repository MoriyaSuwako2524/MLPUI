

import mlpui.model_base as model_base
import torch
import logging
class BASE:
    unet_config = {}
    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
    }

    required_keys = {}
    vae_key_prefix = ["first_stage_model."]
    text_encoder_key_prefix = ["cond_stage_model."]
    supported_inference_dtypes = [torch.float32]
    memory_usage_factor = 2.0
    manual_cast_dtype = None
    custom_operations = None
    quant_config = None  # quantization configuration for mixed precision
    optimizations = {"fp8": False}

    @classmethod
    def matches(s, unet_config, state_dict=None):
        for k in s.unet_config:
            if k not in unet_config or s.unet_config[k] != unet_config[k]:
                return False
        if state_dict is not None:
            for k in s.required_keys:
                if k not in state_dict:
                    return False
        return True

    def model_type(self, state_dict, prefix=""):
        return model_base.ModelType.EPS


    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()

        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.BaseModel(self, model_type=self.model_type(state_dict, prefix), device=device)
        return out



    def process_unet_state_dict(self, state_dict):
        return state_dict


    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.unet_config['dtype'] = dtype
        self.manual_cast_dtype = manual_cast_dtype

    def __getattr__(self, name):
        logging.warning("\nWARNING, you accessed {} from the model config object which doesn't exist. Please fix your code.\n".format(name))
        return None

class UMA(BASE):
    def get_model(self, state_dict, prefix="", device=None):
        import mlpui.model_base as model_base
        out = model_base.UMA(self, device=device)



        return out




models = [UMA]
