"""Internal backend registry. Importing this module does not import model code."""
from .base import ModelBackend
from .newtonnet import NewtonNetBackend
from .torchmdnet import TorchMDNetBackend
from .mace import MACEBackend


_backends = {}
_aliases = {}


def register_backend(backend):
    names = (backend.name, *backend.aliases)
    if len(set(names)) != len(names) or any(name in _aliases for name in names):
        raise ValueError(f"Duplicate backend name or alias: {names}")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Backend names must be nonempty strings")
    _backends[backend.name] = backend
    _aliases.update({name: backend.name for name in names})


def get_backend(name):
    try:
        return _backends[_aliases[name]]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown model family: {name}") from None


def registered_backends():
    return tuple(_backends.values())


def find_backend(state):
    matches = [backend for backend in registered_backends() if backend.matches(state)]
    if len(matches) > 1:
        raise ValueError("Ambiguous model backend")
    return matches[0] if matches else None


register_backend(NewtonNetBackend())
register_backend(TorchMDNetBackend())
register_backend(MACEBackend())
