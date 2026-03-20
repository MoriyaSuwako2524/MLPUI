import yaml
from pathlib import Path


import os

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
CONFIG_DIR = Path(os.environ.get("CHEMLAB_CONFIG_DIR", _DEFAULT_CONFIG_DIR))

_INTERNAL_KEYS = frozenset({"use_defaults"})


class ConfigBase:
    _cache: dict[str, dict] = {}   # filename → loaded data

    yaml_file: str = None   # Default YAML filename; must be set by subclass

    @classmethod
    def _load_yaml(cls, filename: str) -> dict:
        """Load and cache a YAML file relative to CONFIG_DIR."""
        if filename not in cls._cache:
            path = CONFIG_DIR / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Config file not found: {path}\n"
                    f"CONFIG_DIR is set to: {CONFIG_DIR}"
                )
            with open(path, "r", encoding="utf-8") as f:
                cls._cache[filename] = yaml.safe_load(f) or {}
        return cls._cache[filename]

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
    @classmethod
    def _load_path(cls, path: Path) -> dict:
        """Load an arbitrary YAML path (no caching — used for runtime overrides)."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def _resolve_data(cls, raw: dict) -> dict:
        """Apply `use_defaults` inheritance and strip internal keys."""
        data = dict(raw)
        parent_key = data.get("use_defaults")
        if parent_key:
            defaults_all = cls._load_yaml("defaults.yaml")
            parent_data = defaults_all.get(parent_key, {})
            data = {**parent_data, **data}
        return {k: v for k, v in data.items() if k not in _INTERNAL_KEYS}

    @classmethod
    def section_dict(cls) -> dict:
        """Return the resolved config dict from the default yaml_file."""
        if cls.yaml_file is None:
            raise NotImplementedError(f"{cls.__name__} must define `yaml_file`")
        raw = cls._load_yaml(cls.yaml_file)
        return cls._resolve_data(raw)


    def __init__(self, yaml_path: str | Path | None = None):


        if yaml_path is not None:
            raw = self._load_path(Path(yaml_path))
            data = self._resolve_data(raw)
        else:
            data = self.section_dict()

        for k, v in data.items():
            setattr(self, k, v)

    def apply_override(self, overrides: dict):

        _skip = _INTERNAL_KEYS | {"yaml"}

        for key, val in overrides.items():
            if key in _skip or val is None:
                continue
            if not hasattr(self, key):
                continue

            current = getattr(self, key)

            if isinstance(current, list):
                if isinstance(val, str):
                    import ast
                    text = val.strip()
                    if "," in text and not text.startswith("["):
                        text = "[" + text + "]"
                    elif not text.startswith("["):
                        text = "['" + text + "']"
                    try:
                        setattr(self, key, ast.literal_eval(text))
                    except (ValueError, SyntaxError):
                        inner = text.strip("[]")
                        items = [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]
                        setattr(self, key, items)
                else:
                    setattr(self, key, list(val))

            elif isinstance(current, bool):
                if isinstance(val, str):
                    setattr(self, key, val.lower() in ("1", "true", "yes"))
                else:
                    setattr(self, key, bool(val))

            elif isinstance(current, int):
                setattr(self, key, int(val))

            elif isinstance(current, float):
                setattr(self, key, float(val))

            else:
                setattr(self, key, val)


    @classmethod
    def add_to_argparse(cls, parser):
        parser.add_argument(
            "--yaml",
            default=None,
            metavar="PATH",
            help=f"Path to a YAML config file (default: {cls.yaml_file})",
        )

        for key, default in cls.section_dict().items():
            arg = f"--{key}"
            required = (default == "")

            if isinstance(default, bool):
                parser.add_argument(arg, default=None, type=str,
                                    help=f"(default: {default}) [bool]",
                                    required=required)
            elif isinstance(default, int):
                parser.add_argument(arg, default=None, type=int,
                                    help=f"(default: {default}) [int]",
                                    required=required)
            elif isinstance(default, float):
                parser.add_argument(arg, default=None, type=float,
                                    help=f"(default: {default}) [float]",
                                    required=required)
            elif isinstance(default, list):
                parser.add_argument(arg, default=None, type=str,
                                    help=f"(default: {default}) [list]",
                                    required=required)
            else:
                parser.add_argument(arg, default=None,
                                    help=f"(default: '{default}') [str]",
                                    required=required)
        return parser


class QchemEnvConfig(ConfigBase):
    yaml_file = "qchem_env.yaml"