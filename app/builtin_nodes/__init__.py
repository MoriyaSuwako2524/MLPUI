def load_all_nodes():
    from app.builtin_nodes import mlp_loaders
    from app.nodes import register_nodes

    register_nodes(
        mlp_loaders.NODE_CLASS_MAPPINGS,
        mlp_loaders.NODE_DISPLAY_NAME_MAPPINGS,
    )
