"""Built-in nodes for MLPUI — mirrors ComfyUI's node pattern."""

import app.folder_paths as folder_paths


# ── Model loader ─────────────────────────────────────────────────────────────

class LoadMLPModel:
    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("mlp")
        return {
            "required": {
                "model_name": (models or ["(no models found)"],),
                "task": (["omol", "omat", "oc20", "odac", "omc"],),
            }
        }

    RETURN_TYPES = ("MLP_CALCULATOR",)
    RETURN_NAMES = ("calculator",)
    FUNCTION = "load_model"
    CATEGORY = "mlp/loaders"
    DESCRIPTION = "Load a UMA/MLP checkpoint and build an ASE-compatible calculator."

    def load_model(self, model_name: str, task: str):
        from mlpui.calculator import CalculatorBuilder
        ckpt_path = folder_paths.get_full_path("mlp", model_name)
        calc = CalculatorBuilder.from_checkpoint(ckpt_path, task=task).build()
        return (calc,)


# ── Structure builder ─────────────────────────────────────────────────────────

class CreateAtomsFromFormula:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "formula": ("STRING", {"default": "Cu"}),
                "structure": (["fcc", "bcc", "hcp", "sc", "diamond", "molecule"],),
                "a": ("FLOAT", {"default": 3.61, "min": 1.0, "max": 30.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("ASE_ATOMS",)
    RETURN_NAMES = ("atoms",)
    FUNCTION = "create"
    CATEGORY = "mlp/input"
    DESCRIPTION = "Build an ASE Atoms object from a chemical formula."

    def create(self, formula: str, structure: str, a: float):
        if structure == "molecule":
            from ase.build import molecule as ase_molecule
            from ase import Atoms
            try:
                atoms = ase_molecule(formula)
            except Exception:
                atoms = Atoms(formula)
        else:
            from ase.build import bulk
            atoms = bulk(formula, structure, a=a)
        return (atoms,)


# ── Calculator ────────────────────────────────────────────────────────────────

class CalculateEnergyForces:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "calculator": ("MLP_CALCULATOR",),
                "atoms":      ("ASE_ATOMS",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("energy_eV", "summary")
    FUNCTION = "calculate"
    CATEGORY = "mlp/calculate"
    OUTPUT_NODE = True
    DESCRIPTION = "Run the MLP calculator on an Atoms object and return energy & forces."

    def calculate(self, calculator, atoms):
        import copy
        atoms = copy.deepcopy(atoms)
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        n = len(atoms)
        summary = (
            f"Formula : {atoms.get_chemical_formula()}\n"
            f"N atoms : {n}\n"
            f"Energy  : {energy:.6f} eV\n"
            f"E/atom  : {energy / n:.6f} eV\n"
            f"Max |F| : {float(abs(forces).max()):.6f} eV/Å"
        )
        return (energy, summary)


# ── Registry ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LoadMLPModel":           LoadMLPModel,
    "CreateAtomsFromFormula": CreateAtomsFromFormula,
    "CalculateEnergyForces":  CalculateEnergyForces,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMLPModel":           "Load MLP Model",
    "CreateAtomsFromFormula": "Create Atoms",
    "CalculateEnergyForces":  "Calculate Energy & Forces",
}
