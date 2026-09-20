from mlpui.utils import calculate_parameters,_weight_dtype
import logging
import torch
import mlpui.model_patcher
import os
import yaml

def load_checkpoint(ckpt_path=None, state_dict=None, config=None, **kwargs):

    if state_dict is not None:
        raise ValueError("Use a checkpoint file for external models, or load_state_dict_guess_config for UMA")
    model = load_checkpoint_guess_config(ckpt_path, model_config=config, **kwargs)

    return model


def load_checkpoint_guess_config(ckpt_path, output_model=True, model_options=None, disable_dynamic=False,
                                 *, family=None, model_config=None, device=None, dtype=None,
                                 trusted_checkpoint=False):

    from mlpui.external_models import load_external_checkpoint
    model_options = dict(model_options or {})
    dtype = dtype or model_options.get("dtype", model_options.get("weight_dtype"))
    external = load_external_checkpoint(
        ckpt_path, family=family, model_config=model_config, device=device,
        dtype=dtype, trusted_checkpoint=trusted_checkpoint,
    )
    if external is not None:
        return external if output_model else None

    from mlpui.model_loader import load_torch_file

    sd, metadata = load_torch_file(ckpt_path, return_metadata=True)
    model = load_state_dict_guess_config(sd,  output_model, model_options, metadata=metadata, disable_dynamic=disable_dynamic)
    if model is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(ckpt_path, model_detection_error_hint(ckpt_path, sd)))

    return model
def model_detection_error_hint(path, state_dict):
    filename = os.path.basename(path)
    if 'lora' in filename.lower():
        return "\nHINT: This seems to be a Lora file and Lora files should be put in the lora folder and loaded with a lora loader node.."
    return ""


def load_state_dict_guess_config(sd, output_model=True, model_options={}, metadata=None, disable_dynamic=False):
    import mlpui.model_management as model_management
    from mlpui.model_detection import unet_prefix_from_state_dict, model_config_from_unet
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
        ModelPatcher = mlpui.model_patcher.ModelPatcher
        model_patcher = ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
        model.load_model_weights(sd, mlp_model_prefix, assign=model_patcher.is_dynamic())


    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded mlp model directly to GPU")
            model_patcher.to_device(inital_load_device)

    return model_patcher

