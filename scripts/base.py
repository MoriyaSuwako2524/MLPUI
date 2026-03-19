import os
import time
from scripts.config.config_loader import QchemEnvConfig
from typing import Dict
import subprocess


class Script:
    """
    Base class for all scripts used by CLI.

    Subclasses must define:
        name   : CLI subcommand name
        config : a ConfigBase subclass(If the script doesn't require config, config=None)
        run(self, cfg)
    """

    name = None
    config = None

    def run(self, cfg):
        raise NotImplementedError

