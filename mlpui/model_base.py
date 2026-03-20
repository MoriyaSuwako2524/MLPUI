import torch
from enum import Enum
import logging

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

    supported_outputs: tuple[str] = ()


    state_dict_prefix: str = ""

    state_dict_skip_keys: frozenset[str] = frozenset()

    def __init__(self, model_config, device=None):
        super().__init__()
        self.model_config = model_config
        self.device = device
        self.backbone = self._build_backbone(model_config, device)
        self.backbone.eval()

    def _build_backbone(self, model_config, device):

        raise NotImplementedError

    def forward(self, data) -> dict[str, torch.Tensor]:

        raise NotImplementedError

    def load_model_weights(self, sd: dict, assign=False):
        prefix = self.state_dict_prefix
        to_load = {
            k[len(prefix):]: v
            for k, v in sd.items()
            if k.startswith(prefix) and k not in self.state_dict_skip_keys
        }
        to_load = self.model_config.process_state_dict(to_load)
        m, u = self.backbone.load_state_dict(to_load, strict=False, assign=assign)
        if m:
            logging.warning(f"{self.__class__.__name__} missing keys: {m}")
        if u:
            logging.warning(f"{self.__class__.__name__} unexpected keys: {u}")
        return self

    def get_output(self, data, outputs: list[str] | None = None) -> dict[str, torch.Tensor]:

        result = self.forward(data)
        if outputs is None:
            return result
        return {k: result[k] for k in outputs if k in result}

    def memory_required(self, num_atoms: int) -> int:
        """子类可覆盖，估算显存需求。"""
        raise NotImplementedError