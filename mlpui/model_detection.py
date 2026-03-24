
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

def detect_unet_config(state_dict, key_prefix, metadata=None):
    keys = list(state_dict.keys())
    if "{}blocks.0.edge_wise.so2_conv_1.fc_m0.weights".format(key_prefix) in keys:

        mlp_config = {}
        mlp_config["model_name"] = "uma"
        mlp_config["num_layers"] = count_blocks(keys, "{}blocks.{}".format(key_prefix, "{}"))

        sphere_emb_key = "{}sphere_embedding.weight".format(key_prefix)
        if sphere_emb_key in state_dict:
            mlp_config["max_num_elements"] = state_dict[sphere_emb_key].shape[0]
            mlp_config["sphere_channels"] = state_dict[sphere_emb_key].shape[1]

        src_emb_key = "{}source_embedding.weight".format(key_prefix)
        if src_emb_key in state_dict:
            mlp_config["edge_channels"] = state_dict[src_emb_key].shape[1]

        rad_key = "{}edge_degree_embedding.rad_func.net.0.weight".format(key_prefix)
        if rad_key in state_dict and "edge_channels" in mlp_config:
            input_dim = state_dict[rad_key].shape[1]
            mlp_config["num_distance_basis"] = input_dim - 2 * mlp_config["edge_channels"]

        grid_mlp_key = "{}blocks.0.atom_wise.grid_mlp.0.weight".format(key_prefix)
        scalar_mlp_key = "{}blocks.0.atom_wise.scalar_mlp.0.weight".format(key_prefix)
        if grid_mlp_key in state_dict:
            mlp_config["hidden_channels"] = state_dict[grid_mlp_key].shape[0]
        elif scalar_mlp_key in state_dict:
            mlp_config["hidden_channels"] = state_dict[scalar_mlp_key].shape[0]


        if "{}blocks.0.atom_wise.scalar_mlp.0.weight".format(key_prefix) in keys:
            mlp_config["ff_type"] = "spectral"
        else:
            mlp_config["ff_type"] = "grid"

        if "{}charge_embedding.rand_emb.weight".format(key_prefix) in keys:
            mlp_config["chg_spin_emb_type"] = "rand_emb"
        elif "{}charge_embedding.lin_emb.weight".format(key_prefix) in keys:
            mlp_config["chg_spin_emb_type"] = "lin_emb"
        else:
            mlp_config["chg_spin_emb_type"] = "pos_emb"


        mlp_config["has_mole"] = "{}routing_mlp.0.weight".format(key_prefix) in keys
        datasets = [
            k.split(".")[-2]
            for k in keys
            if "dataset_emb_dict" in k and k.endswith(".weight")
        ]
        if datasets:
            mlp_config["dataset_mapping"] = {name: name for name in datasets}
            mlp_config["use_dataset_embedding"] = True

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