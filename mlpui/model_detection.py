








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

def model_config_from_unet(state_dict, key_prefix):
    keys = list(state_dict.keys())
    if "{}blocks.0.edge_wise.so2_conv_1.fc_m0.weights".format(key_prefix) in keys:

        mlp_config = {}
        mlp_config["model_type"] = "uma"
        mlp_config["num_blocks"] = count_blocks(keys, "{}blocks.{}".format(key_prefix, "{}"))
        mlp_config["has_mole"] = "{}routing_mlp.0.weight".format(key_prefix) in keys
        mlp_config["datasets"] = [
            k.split(".")[-2]  # omol, omat, odac ...
            for k in keys
            if "dataset_emb_dict" in k and k.endswith(".weight")
        ]
        return mlp_config

    return None