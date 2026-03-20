from mlpui.models.model_loader import load_torch_file





def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}, disable_dynamic=False):
    # TODO: Modify this function to load mlp state dict
    sd, metadata = load_torch_file(ckpt_path, return_metadata=True)
    out = load_state_dict_guess_config(sd, output_vae, output_clip, output_clipvision, embedding_directory, output_model, model_options, te_model_options=te_model_options, metadata=metadata, disable_dynamic=disable_dynamic)
    if out is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(ckpt_path, model_detection_error_hint(ckpt_path, sd)))
    if output_model and out[0] is not None:
        out[0].cached_patcher_init = (load_checkpoint_guess_config_model_only, (ckpt_path, embedding_directory, model_options, te_model_options))
    if output_clip and out[1] is not None:
        out[1].patcher.cached_patcher_init = (load_checkpoint_guess_config_clip_only, (ckpt_path, embedding_directory, model_options, te_model_options))
    return out



def load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}, metadata=None, disable_dynamic=False):
    #TODO: Modify this function to load mlp state dict
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype = comfy.utils.weight_dtype(sd, diffusion_model_prefix)
    load_device = model_management.get_torch_device()

    custom_operations = model_options.get("custom_operations", None)
    if custom_operations is None:
        sd, metadata = comfy.utils.convert_old_quants(sd, diffusion_model_prefix, metadata=metadata)

    model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix, metadata=metadata)
    if model_config is None:
        logging.warning("Warning, This is not a checkpoint file, trying to load it as a diffusion model only.")
        diffusion_model = load_diffusion_model_state_dict(sd, model_options={})
        if diffusion_model is None:
            return None
        return (diffusion_model, None, VAE(sd={}), None)  # The VAE object is there to throw an exception if it's actually used'

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    if custom_operations is not None:
        model_config.custom_operations = custom_operations

    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

    if unet_dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)

    if model_config.quant_config is not None:
        manual_cast_dtype = model_management.unet_manual_cast(None, load_device, model_config.supported_inference_dtypes)
    else:
        manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
        ModelPatcher = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
        model_patcher = ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
        model.load_model_weights(sd, diffusion_model_prefix, assign=model_patcher.is_dynamic())

    if output_vae:
        vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd, metadata=metadata)

    if output_clip:
        if te_model_options.get("custom_operations", None) is None:
            scaled_fp8_list = []
            for k in list(sd.keys()):  # Convert scaled fp8 to mixed ops
                if k.endswith(".scaled_fp8"):
                    scaled_fp8_list.append(k[:-len("scaled_fp8")])

            if len(scaled_fp8_list) > 0:
                out_sd = {}
                for k in sd:
                    skip = False
                    for pref in scaled_fp8_list:
                        skip = skip or k.startswith(pref)
                    if not skip:
                        out_sd[k] = sd[k]

                for pref in scaled_fp8_list:
                    quant_sd, qmetadata = comfy.utils.convert_old_quants(sd, pref, metadata={})
                    for k in quant_sd:
                        out_sd[k] = quant_sd[k]
                    sd = out_sd

        clip_target = model_config.clip_target(state_dict=sd)
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                parameters = comfy.utils.calculate_parameters(clip_sd)
                clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=parameters, state_dict=clip_sd, model_options=te_model_options, disable_dynamic=disable_dynamic)
            else:
                logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded diffusion model directly to GPU")
            model_management.load_models_gpu([model_patcher], force_full_load=True)

    return (model_patcher, clip, vae, clipvision)