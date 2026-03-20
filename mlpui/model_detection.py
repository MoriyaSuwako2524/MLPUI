
import mlpui.supported_models as supported_models
import logging






def unet_prefix_from_state_dict(state_dict):
    candidates = ["model.diffusion_model.",
                  "model.model.",
                  "net.",
                  "module.backbone.", #uma fairchem
                  ]
    counts = {k: 0 for k in candidates}
    for k in state_dict:
        for c in candidates:
            if k.startswith(c):
                counts[c] += 1
                break

    top = max(counts, key=counts.get)
    if counts[top] > 5:
        return top
    else:
        return "model." # others

def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count

def detect_unet_config(state_dict, key_prefix,metadata=None):
    keys = list(state_dict.keys())
    if "{}blocks.0.edge_wise.so2_conv_1.fc_m0.weights".format(key_prefix) in keys:

        mlp_config = {}
        mlp_config["model_name"] = "uma"
        mlp_config["num_blocks"] = count_blocks(keys, "{}blocks.{}".format(key_prefix, "{}"))
        mlp_config["has_mole"] = "{}routing_mlp.0.weight".format(key_prefix) in keys
        mlp_config["datasets"] = [
            k.split(".")[-2]  # omol, omat, odac ...
            for k in keys
            if "dataset_emb_dict" in k and k.endswith(".weight")
        ]
        return mlp_config

    return None


def model_config_from_unet_config(unet_config, state_dict=None):
    for model_config in supported_models.models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)

    logging.error("no match {}".format(unet_config))
    return None

def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None):
    unet_config = detect_unet_config(state_dict, unet_key_prefix, metadata=metadata)
    if unet_config is None:
        return None


    model_config = model_config_from_unet_config(unet_config, state_dict)

    if model_config is None and use_base_if_no_match:
        model_config = supported_models.BASE(unet_config)

    return model_config