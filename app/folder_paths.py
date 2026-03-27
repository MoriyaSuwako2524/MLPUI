import os

# Project root  (parent of app/)
_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models_dir = os.path.join(_base_path, "models")


_MODEL_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors"}


def get_filename_list(folder_name: str) -> list[str]:
    folder = os.path.join(models_dir, folder_name)
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in _MODEL_EXTENSIONS
    )


def get_full_path(folder_name: str, filename: str) -> str:
    return os.path.join(models_dir, folder_name, filename)
