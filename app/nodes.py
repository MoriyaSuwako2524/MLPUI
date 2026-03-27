"""Node registry — mirrors ComfyUI's node system."""

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def register_nodes(mappings: dict, display_names: dict | None = None) -> None:
    NODE_CLASS_MAPPINGS.update(mappings)
    if display_names:
        NODE_DISPLAY_NAME_MAPPINGS.update(display_names)


def get_object_info() -> dict:
    """Return all node definitions as a JSON-serialisable dict.

    Output format mirrors ComfyUI's /object_info response so the same
    frontend registration logic works for both.
    """
    result = {}
    for node_id, node_class in NODE_CLASS_MAPPINGS.items():
        raw_inputs = node_class.INPUT_TYPES()

        # Normalise each section to {name: [type_or_list, config_dict]}
        processed: dict = {}
        for section in ("required", "optional"):
            entries = raw_inputs.get(section)
            if not entries:
                continue
            processed[section] = {}
            for name, spec in entries.items():
                if isinstance(spec, (list, tuple)):
                    type_info = spec[0]
                    config = spec[1] if len(spec) > 1 else {}
                else:
                    type_info = spec
                    config = {}
                processed[section][name] = [type_info, config]

        return_types = list(node_class.RETURN_TYPES)
        return_names = list(getattr(node_class, "RETURN_NAMES", return_types))

        result[node_id] = {
            "input": processed,
            "output": return_types,
            "output_name": return_names,
            "name": node_id,
            "display_name": NODE_DISPLAY_NAME_MAPPINGS.get(
                node_id, node_id.replace("_", " ")
            ),
            "description": getattr(node_class, "DESCRIPTION", ""),
            "category": getattr(node_class, "CATEGORY", "default"),
            "output_node": getattr(node_class, "OUTPUT_NODE", False),
        }

    return result
