import torch
from enum import Enum
import logging

import mlpui.supported_models as supported_models
import mlpui.models.uma.uma as uma
import mlpui.model_management as model_management

class ModelType(Enum):
    #TODO: Shift to GNN type?
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5
    FLOW = 6
    V_PREDICTION_CONTINUOUS = 7
    FLUX = 8
    IMG_TO_IMG = 9
    FLOW_COSMOS = 10
    IMG_TO_IMG_FLOW = 11



class BaseModel(torch.nn.Module):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None, unet_model=uma.UMA):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        self.current_patcher: 'ModelPatcher' = None

        if not unet_config.get("disable_unet_model_creation", False):
            self.diffusion_model = unet_model(**unet_config, device=device)
            self.diffusion_model.eval()
            logging.info("model weight dtype {}, manual cast: {}".format(self.get_dtype(), self.manual_cast_dtype))
            model_management.archive_model_dtypes(self.diffusion_model)

        self.model_type = model_type

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0

        self.concat_keys = ()
        logging.info("model_type {}".format(model_type.name))
        logging.debug("adm {}".format(self.adm_channels))
        self.memory_usage_factor = model_config.memory_usage_factor
        self.memory_usage_factor_conds = ()
        self.memory_usage_shape_process = {}

class UMA(BaseModel):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=uma.UMA)

