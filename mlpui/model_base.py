import os
import yaml
import torch
from enum import Enum
import logging

import mlpui.supported_models as supported_models
import mlpui.models.uma.uma as uma
import mlpui.model_management as model_management
import inspect


def _find_elem_refs_yaml() -> str | None:
    """Find iso_atom_elem_refs.yaml relative to the mlpui package."""
    pkg_dir = os.path.dirname(__file__)
    path = os.path.normpath(os.path.join(pkg_dir, "..", "models", "mlp", "iso_atom_elem_refs.yaml"))
    return path if os.path.exists(path) else None


def _load_elem_refs_from_yaml() -> dict[str, list[float]]:
    """Load per-dataset element references from iso_atom_elem_refs.yaml.

    The file contains entries like ``oc20_elem_refs`` (list indexed by Z) and
    ``omol_elem_refs`` (dict {Z: {charge: energy}}).  All are converted to a
    flat list indexed by atomic number so they can be used directly as
    ``elem_refs[atomic_numbers]`` in EnergyAndForceHead.
    """
    path = _find_elem_refs_yaml()
    if path is None:
        logging.warning("iso_atom_elem_refs.yaml not found; no YAML element references loaded.")
        return {}

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    result: dict[str, list[float]] = {}
    for key, val in data.items():
        if not key.endswith("_elem_refs"):
            continue
        dataset = key[: -len("_elem_refs")]  # "oc20_elem_refs" → "oc20"
        if isinstance(val, list):
            result[dataset] = [float(v) for v in val]
        elif isinstance(val, dict):
            # omol format: {Z (int): {charge (int): energy (float)}}
            max_z = max(int(z) for z in val.keys()) + 1
            size = max(max_z, 120)
            refs = [0.0] * size
            for z_str, charge_dict in val.items():
                z = int(z_str)
                if isinstance(charge_dict, dict):
                    # Use neutral (charge=0) reference
                    energy = charge_dict.get(0, charge_dict.get("0", None))
                    if energy is not None:
                        refs[z] = float(energy)
                elif isinstance(charge_dict, (int, float)):
                    refs[z] = float(charge_dict)
            result[dataset] = refs

    logging.info("Loaded element references from YAML for datasets: %s", list(result.keys()))
    return result


def _load_normalizer_into_head(head, metadata: dict) -> None:
    """Parse tasks_config from checkpoint metadata and populate head normaliser.

    Element references are loaded from iso_atom_elem_refs.yaml first (primary
    source), then overridden by any refs successfully parsed from tasks_config.
    """
    rmsd = 1.0
    mean = 0.0

    # Primary source: YAML files ship with the model directory
    dataset_refs: dict[str, list[float]] = _load_elem_refs_from_yaml()

    tasks_config = metadata.get("tasks_config", None) if metadata else None
    if tasks_config:
        for task in tasks_config:
            if task.get("property") != "energy":
                continue
            norm = task.get("normalizer", {})
            rmsd = float(norm.get("rmsd", rmsd))
            mean = float(norm.get("mean", mean))

            # If tasks_config carries element refs, let them override the YAML
            elem_refs_cfg = task.get("element_references")
            if elem_refs_cfg:
                try:
                    refs_list = elem_refs_cfg["element_references"]["_args_"][0]
                except (KeyError, IndexError, TypeError):
                    refs_list = None
                if refs_list is not None:
                    for dataset in task.get("datasets", []):
                        dataset_refs[dataset] = refs_list

    head.load_normalizer(rmsd=rmsd, mean=mean, dataset_element_refs=dataset_refs)
    logging.info("Loaded normaliser rmsd=%.4f, mean=%.4f for datasets=%s",
                 rmsd, mean, list(dataset_refs.keys()))
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
    ENERGY_FORCE = 12
def filter_kwargs_for_class(cls, config: dict) -> dict:
    valid_params = set()
    has_var_keyword = False

    for klass in cls.__mro__:
        if klass is object:
            continue
        sig = inspect.signature(klass.__init__)
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:

                continue
            valid_params.add(name)

    return {k: v for k, v in config.items() if k in valid_params}

class BaseModel(torch.nn.Module):
    def __init__(self, model_config, model_type=ModelType.ENERGY_FORCE, device=None, unet_model=uma.eSCNMDBackbone):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        self.current_patcher: 'ModelPatcher' = None

        if not unet_config.get("disable_unet_model_creation", False):
            sig = inspect.signature(unet_model.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}

            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )


            filtered_config  = filter_kwargs_for_class(unet_model, unet_config)

            self.diffusion_model = unet_model(**filtered_config)
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

    def load_model_weights(self, sd, unet_prefix="", assign=False):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix):]] = sd.pop(k)

        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False, assign=assign)
        if len(m) > 0:
            logging.warning("unet missing: {}".format(m))

        if len(u) > 0:
            logging.warning("unet unexpected: {}".format(u))
        del to_load
        return self

class UMA(BaseModel):
    def __init__(self, model_config, model_type=ModelType.ENERGY_FORCE, device=None):
        if model_config.unet_config.get("has_mole", False):
            unet_cls = uma.eSCNMDMoeBackbone
        else:
            unet_cls = uma.eSCNMDBackbone
        print("unet_config:", model_config.unet_config)
        super().__init__(model_config, model_type, device=device, unet_model=unet_cls)
        self.output_head = None

    def get_dtype(self):
        try:
            return next(self.diffusion_model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def load_model_weights(self, sd, unet_prefix="", assign=False, metadata=None):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix):]] = sd.pop(k)
        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False, assign=assign)
        if len(m) > 0:
            logging.warning("unet missing: {}".format(m))
        if len(u) > 0:
            logging.warning("unet unexpected: {}".format(u))

        # Load output head weights from the remaining state dict.
        # Head keys are NOT under unet_prefix (e.g. "module.backbone.") but
        # under a sibling prefix (e.g. "module.output_heads.*"), so they are
        # still present in `sd` after the backbone keys were popped out.
        HEAD_SUFFIX = "output_heads.energyandforcehead.head."
        head_sd = {}
        for k in list(sd.keys()):
            # Find the segment after the last occurrence of HEAD_SUFFIX
            idx = k.find(HEAD_SUFFIX)
            if idx != -1:
                short_key = k[idx + len(HEAD_SUFFIX):]
                head_sd[short_key] = sd.pop(k)
        if head_sd:
            from mlpui.models.uma.output_heads import EnergyAndForceHead
            sphere_channels = self.model_config.unet_config.get("sphere_channels", 128)
            head = EnergyAndForceHead(sphere_channels=sphere_channels)
            missing, unexpected = head.load_state_dict(head_sd, strict=False)
            if missing:
                logging.warning("output_head missing keys: {}".format(missing))
            if unexpected:
                logging.warning("output_head unexpected keys: {}".format(unexpected))
            _load_normalizer_into_head(head, metadata or {})
            self.output_head = head
            logging.info("Loaded EnergyAndForceHead (sphere_channels=%d)", sphere_channels)

        del to_load
        return self


