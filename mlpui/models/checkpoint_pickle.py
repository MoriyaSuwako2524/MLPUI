"""Legacy module-name migration, used ONLY for explicitly trusted pickle files.

This does not make pickle safe. State dictionaries still use weights_only=True.
No global sys.modules aliases or installed backend packages are needed.
"""
import pickle
from pickle import dump, dumps, load, loads  # torch's pickle-module interface


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        for family in ("newtonnet", "torchmdnet"):
            if module == family or module.startswith(family + "."):
                module = "mlpui.models." + module
                break
        return super().find_class(module, name)
