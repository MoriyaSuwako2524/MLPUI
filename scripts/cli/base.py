import pkgutil
import importlib
import inspect
from types import ModuleType

import scripts
from scripts.base import Script


class CLICommand:

    name = None

    def add_arguments(self, parser, subparsers):

        for script_cls in discover_scripts():

            group, command = script_to_cli(script_cls)

            if group != self.name:
                continue

            # Create parser for the script
            p = subparsers.add_parser(command, help=script_cls.__doc__)

            # Add config parameters if the script uses a config class
            cfg_class = getattr(script_cls, "config", None)
            if cfg_class:
                cfg_class.add_to_argparse(p)

            script_obj = script_cls()

            # Bind function
            p.set_defaults(
                func=lambda args,
                             script=script_obj,
                             cfg_class=cfg_class:
                    self.run_script(script, cfg_class, args)
            )

    def run_script(self, script, cfg_class, args):
        """Load cfg (if needed) and run the script."""
        if cfg_class:
            cfg = cfg_class()
            cfg.apply_override(vars(args))
        else:
            cfg = None

        return script.run(cfg)

    def register(self, top_subparsers):
        """Register this CLI group under top-level CLI."""
        parser = top_subparsers.add_parser(self.name)
        subparsers = parser.add_subparsers(dest=f"{self.name}_cmd")
        self.add_arguments(parser, subparsers)



_DISCOVER_CACHE = None
def discover_scripts():

    global _DISCOVER_CACHE
    if _DISCOVER_CACHE is not None:
        return _DISCOVER_CACHE
    scripts = []

    for _, modname, _ in pkgutil.walk_packages(
        chemlab.scripts.__path__, chemlab.scripts.__name__ + "."
    ):
        module = importlib.import_module(modname)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Script) and obj is not Script:
                scripts.append(obj)
    _DISCOVER_CACHE = scripts
    return scripts



def script_to_cli(script_cls):
    module = script_cls.__module__
    parts = module.split(".")
    group = parts[-2]
    command = parts[-1]
    return group, command
