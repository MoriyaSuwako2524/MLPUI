
import mlpui.supported_models as supported_models
import logging


# Parameters accepted by eSCNMDBackbone that we want to extract from model_config.
# These are the ones that cannot be reliably inferred from state dict key shapes.
_BACKBONE_KEYS = {
    "lmax", "mmax", "cutoff", "max_neighbors", "norm_type", "act_type",
    "sphere_channels", "hidden_channels", "edge_channels", "num_layers",
    "max_num_elements", "num_distance_basis", "ff_type", "chg_spin_emb_type",
    "has_mole", "num_experts", "use_composition_embedding", "use_dataset_embedding",
    "dataset_mapping", "dataset_list", "use_quaternion_wigner",
    "charge_balanced_channels", "spin_balanced_channels",
    "execution_mode", "cs_emb_grad", "dataset_emb_grad",
    "activation_checkpointing", "edge_chunk_size",
}


def _extract_backbone_config(cfg, depth: int = 0) -> dict:
    """Recursively search an OmegaConf/dict config for eSCNMDBackbone parameters.

    Returns the sub-dict that contains the most backbone-relevant keys.
    """
    if depth > 8 or not hasattr(cfg, "items"):
        return {}

    # Collect keys at this level that match backbone params
    local = {}
    for k, v in cfg.items():
        if k in _BACKBONE_KEYS:
            # Unwrap OmegaConf primitives to plain Python types
            try:
                from omegaconf import OmegaConf
                local[k] = OmegaConf.to_object(v) if hasattr(v, "_metadata") else v
            except Exception:
                local[k] = v

    # Recurse into sub-dicts and keep the best match
    best_sub = {}
    for k, v in cfg.items():
        if hasattr(v, "items"):
            sub = _extract_backbone_config(v, depth + 1)
            if len(sub) > len(best_sub):
                best_sub = sub

    # Prefer deeper node if it has more backbone keys; merge with local
    if len(best_sub) >= len(local):
        # Sub-node wins; fill any gaps with local
        merged = dict(local)
        merged.update(best_sub)
        return merged
    return local


def _backbone_config_from_metadata(metadata: dict) -> dict:
    """Extract eSCNMDBackbone config from checkpoint metadata["model_config"]."""
    if not metadata:
        return {}
    model_config = metadata.get("model_config", None)
    if model_config is None:
        return {}
    try:
        result = _extract_backbone_config(model_config)
        if result:
            logging.info("Extracted backbone config from checkpoint model_config: %s",
                         {k: v for k, v in result.items() if k in {"lmax", "mmax", "cutoff", "sphere_channels", "num_layers"}})
        return result
    except Exception as e:
        logging.warning("Failed to extract backbone config from metadata: %s", e)
        return {}


def _detect_lmax_mmax_from_state_dict(state_dict, key_prefix) -> dict:
    """Detect lmax and mmax from state dict as a fallback.

    lmax: count Jd_{l} buffers (registered for l=0..lmax).
    mmax: count fc_m{m} groups in so2_conv_1 of the first block.
    """
    result = {}
    keys = list(state_dict.keys())

    # lmax from Jd buffers
    jd_ls = []
    for k in keys:
        if k.startswith(f"{key_prefix}Jd_"):
            tail = k[len(f"{key_prefix}Jd_"):]
            l_str = tail.split(".")[0]
            if l_str.isdigit():
                jd_ls.append(int(l_str))
    if jd_ls:
        result["lmax"] = max(jd_ls)

    # mmax from so2_conv fc_m groups
    fc_ms = []
    prefix_so2 = f"{key_prefix}blocks.0.edge_wise.so2_conv_1.fc_m"
    for k in keys:
        if k.startswith(prefix_so2):
            tail = k[len(prefix_so2):]
            m_str = tail.split(".")[0]
            if m_str.isdigit():
                fc_ms.append(int(m_str))
    if fc_ms:
        result["mmax"] = max(fc_ms)

    return result


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

        # ── Step 1: extract from checkpoint model_config (most reliable) ──────
        meta_cfg = _backbone_config_from_metadata(metadata)
        mlp_config.update(meta_cfg)

        # ── Step 2: state-dict detection fills or verifies individual keys ──────

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

        # Detect lmax FIRST — needed to correctly decode hidden_channels below.
        # norm.affine_weight: [lmax+1, sphere_channels]
        # so3_linear_1.weight: [lmax+1, hidden_channels, sphere_channels]
        norm_key = "{}norm.affine_weight".format(key_prefix)
        so3_key  = "{}blocks.0.atom_wise.so3_linear_1.weight".format(key_prefix)
        if norm_key in state_dict:
            mlp_config["lmax"] = state_dict[norm_key].shape[0] - 1
        elif so3_key in state_dict:
            mlp_config["lmax"] = state_dict[so3_key].shape[0] - 1

        # Detect ff_type before hidden_channels (the formula differs).
        scalar_mlp_key = "{}blocks.0.atom_wise.scalar_mlp.0.weight".format(key_prefix)
        grid_mlp_key   = "{}blocks.0.atom_wise.grid_mlp.0.weight".format(key_prefix)
        if scalar_mlp_key in keys:
            mlp_config["ff_type"] = "spectral"
        else:
            mlp_config["ff_type"] = "grid"

        # Detect hidden_channels.
        # For grid ff_type:    grid_mlp.0.weight = [hidden_channels, sphere_channels]
        # For spectral ff_type: scalar_mlp.0.weight = [lmax * hidden_channels, sphere_channels]
        if grid_mlp_key in state_dict:
            mlp_config["hidden_channels"] = state_dict[grid_mlp_key].shape[0]
        elif scalar_mlp_key in state_dict:
            raw = state_dict[scalar_mlp_key].shape[0]
            lmax = mlp_config.get("lmax", 1)
            mlp_config["hidden_channels"] = raw // lmax if lmax > 0 else raw

        if "{}charge_embedding.rand_emb.weight".format(key_prefix) in keys:
            mlp_config["chg_spin_emb_type"] = "rand_emb"
        elif "{}charge_embedding.lin_emb.weight".format(key_prefix) in keys:
            mlp_config["chg_spin_emb_type"] = "lin_emb"
        else:
            mlp_config.setdefault("chg_spin_emb_type", "pos_emb")

        mlp_config["has_mole"] = "{}routing_mlp.0.weight".format(key_prefix) in keys
        if mlp_config["has_mole"]:
            mole_weight_key = "{}blocks.0.edge_wise.so2_conv_1.fc_m0.weights".format(key_prefix)
            if mole_weight_key in state_dict and state_dict[mole_weight_key].ndim == 3:
                mlp_config["num_experts"] = state_dict[mole_weight_key].shape[0]

        mlp_config["use_composition_embedding"] = "{}composition_embedding.weight".format(key_prefix) in keys

        # ── Step 3: mmax detection from state dict; lmax already set above ──────
        # mmax: count fc_m{m} groups in so2_conv_1 of block 0
        lm = _detect_lmax_mmax_from_state_dict(state_dict, key_prefix)
        if "mmax" not in mlp_config:
            mlp_config["mmax"] = lm.get("mmax", mlp_config.get("lmax", 2))
        # lmax: state-dict shape detection (Step 2) is authoritative;
        # fall back to Jd buffers only if Step 2 yielded nothing
        if "lmax" not in mlp_config:
            mlp_config["lmax"] = lm.get("lmax", 2)

        # ── Step 4: dataset embedding from state dict ──────────────────────────
        datasets = [
            k.split(".")[-2]
            for k in keys
            if "dataset_emb_dict" in k and k.endswith(".weight")
        ]
        if datasets:
            mlp_config["dataset_mapping"] = {name: name for name in datasets}
            mlp_config["use_dataset_embedding"] = True
        # dataset_list is deprecated; remove it when dataset_mapping is present
        # to avoid the "both provided" conflict in resolve_dataset_mapping().
        if "dataset_mapping" in mlp_config and "dataset_list" in mlp_config:
            del mlp_config["dataset_list"]

        # Output head detection
        heads = {}
        head_prefix = "module.output_heads."
        for key in state_dict.keys():
            if not key.startswith(head_prefix):
                continue
            remainder = key[len(head_prefix):]
            parts = remainder.split(".", 1)
            if len(parts) < 2:
                continue
            head_name = parts[0]
            if head_name not in heads:
                heads[head_name] = {
                    "type": "unknown",
                    "has_mole": False,
                    "energy_block_layers": [],
                }
            sub_key = parts[1]
            if "energy_block" in sub_key:
                heads[head_name]["type"] = "efs"
            if "global_mole_tensors" in sub_key or sub_key.startswith("head.") and "weights" in sub_key:
                if state_dict[key].ndim == 3:
                    heads[head_name]["has_mole"] = True

        for head_name, config in heads.items():
            if config["type"] != "efs":
                continue
            layers = {}
            for key in state_dict.keys():
                block_prefix = f"{head_prefix}{head_name}.head.energy_block."
                if not key.startswith(block_prefix):
                    continue
                sub = key[len(block_prefix):]
                parts = sub.split(".")
                layer_idx = int(parts[0])
                param_type = parts[1]
                if param_type == "weight":
                    w = state_dict[key]
                    layers[layer_idx] = {"out": w.shape[0], "in": w.shape[1]}
            config["energy_block_layers"] = [layers[k] for k in sorted(layers.keys())]

        logging.info("Detected UMA config: lmax=%s mmax=%s sphere_channels=%s num_layers=%s",
                     mlp_config.get("lmax"), mlp_config.get("mmax"),
                     mlp_config.get("sphere_channels"), mlp_config.get("num_layers"))
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
