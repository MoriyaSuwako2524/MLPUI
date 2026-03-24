from mlpui.model_loader import load_torch_file
from mlpui.utils import calculate_parameters,_weight_dtype
import mlpui.model_management as model_management
from model_detection import unet_prefix_from_state_dict,model_config_from_unet
import logging
import torch
import mlpui.model_patcher
def load_checkpoint_guess_config(ckpt_path,  output_model=True, model_options={}, disable_dynamic=False):
    # TODO: Modify this function to load mlp state dict
    sd, metadata = load_torch_file(ckpt_path, return_metadata=True)
    out = load_state_dict_guess_config(sd,  output_model, model_options, metadata=metadata, disable_dynamic=disable_dynamic)
    if out is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(ckpt_path, model_detection_error_hint(ckpt_path, sd)))

    return out



def load_state_dict_guess_config(sd, output_model=True, model_options={}, metadata=None, disable_dynamic=False):
    #TODO:

    model = None
    model_patcher = None

    mlp_model_prefix = unet_prefix_from_state_dict(sd)
    parameters = calculate_parameters(sd, mlp_model_prefix)
    weight_dtype = _weight_dtype(sd, mlp_model_prefix)
    load_device = model_management.get_torch_device()

    model_config = model_config_from_unet(sd, mlp_model_prefix, metadata=metadata)
    if model_config is None:
        logging.warning("Warning, This is not a checkpoint file")
        return None

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

    if unet_dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        model = model_config.get_model(sd, mlp_model_prefix, device=inital_load_device)
        ModelPatcher = model_patcher.ModelPatcher if disable_dynamic else model_patcher.CoreModelPatcher
        model_patcher = ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
        model.load_model_weights(sd, mlp_model_prefix, assign=model_patcher.is_dynamic())


    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded mlp model directly to GPU")
            model_management.load_models_gpu([model_patcher], force_full_load=True)

    return model_patcher

