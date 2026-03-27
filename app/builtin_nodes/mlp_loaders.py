"""Built-in nodes for MLPUI — mirrors ComfyUI's node pattern."""

import copy
import numpy as np
import app.folder_paths as folder_paths

# ─── Unit conversion tables ──────────────────────────────────────────────────

_ENERGY_CONV = {
    "eV":       1.0,
    "kcal/mol": 23.0605,
    "kJ/mol":   96.4853,
    "hartree":  0.036749,
}

_FORCE_CONV = {
    "eV/Å":         1.0,
    "kcal/mol/Å":   23.0605,
    "kJ/mol/Å":     96.4853,
    "hartree/bohr": 0.019447,
}


# ─── Result container ────────────────────────────────────────────────────────

class MLPResult:
    """Container returned by LoadMLPModel; carries atoms + computed properties."""
    __slots__ = ("atoms", "energy", "forces", "formula", "n_atoms")

    def __init__(self, atoms, energy, forces):
        self.atoms   = atoms
        self.energy  = float(energy)
        self.forces  = np.asarray(forces)
        self.formula = atoms.get_chemical_formula()
        self.n_atoms = len(atoms)


# ─── Input nodes ─────────────────────────────────────────────────────────────

class ReadXYZFile:
    @classmethod
    def INPUT_TYPES(cls):
        xyz_files = folder_paths.get_xyz_list()
        placeholder = ["(drop an .xyz file onto the canvas)"]
        return {
            "required": {
                "filepath": (xyz_files if xyz_files else placeholder,),
            },
            "optional": {
                "charge": ("INT", {"default": 0, "min": -20, "max": 20}),
                "spin":   ("INT", {"default": 1, "min": 1,  "max": 20}),
                "index":  ("INT", {"default": 0, "min": 0,  "max": 99999}),
            },
        }

    RETURN_TYPES = ("ASE_ATOMS",)
    RETURN_NAMES = ("atoms",)
    FUNCTION     = "read"
    CATEGORY     = "mlp/input"
    DESCRIPTION  = "Read XYZ file. charge = total charge; spin = multiplicity (2S+1)."

    def read(self, filepath: str, charge: int = 0, spin: int = 1, index: int = 0):
        from ase.io import read
        full_path = folder_paths.get_xyz_path(filepath)
        atoms = read(full_path, index=index)
        # UMAInputAdapter reads charge/spin from atoms.info, not initial_charges.
        # spin widget = multiplicity (2S+1); model expects 2S.
        atoms.info["charge"] = charge
        atoms.info["spin"]   = spin - 1   # 2S
        return (atoms,)


class ReadXYZBatch:
    @classmethod
    def INPUT_TYPES(cls):
        xyz_files = folder_paths.get_xyz_list()
        placeholder = ["(drop an .xyz file onto the canvas)"]
        return {
            "required": {
                "filepath": (xyz_files if xyz_files else placeholder,),
            },
            "optional": {
                "charge": ("INT", {"default": 0, "min": -20, "max": 20}),
                "spin":   ("INT", {"default": 1, "min": 1,  "max": 20}),
            },
        }

    RETURN_TYPES = ("ASE_ATOMS_LIST",)
    RETURN_NAMES = ("atoms_list",)
    FUNCTION     = "read"
    CATEGORY     = "mlp/input"
    DESCRIPTION  = "Read ALL frames from a multi-frame XYZ file (index=':')."

    def read(self, filepath: str, charge: int = 0, spin: int = 1):
        from ase.io import read
        full_path = folder_paths.get_xyz_path(filepath)
        atoms_list = read(full_path, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        for atoms in atoms_list:
            atoms.info["charge"] = charge
            atoms.info["spin"]   = spin - 1
        return (atoms_list,)


class CreateAtomsFromFormula:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "formula":   ("STRING", {"default": "Cu"}),
                "structure": (["fcc", "bcc", "hcp", "sc", "diamond", "molecule"],),
                "a":         ("FLOAT",  {"default": 3.61, "min": 1.0, "max": 30.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("ASE_ATOMS",)
    RETURN_NAMES = ("atoms",)
    FUNCTION     = "create"
    CATEGORY     = "mlp/input"
    DESCRIPTION  = "Build an ASE Atoms object from a chemical formula (no file needed)."

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


# ─── Inference node ──────────────────────────────────────────────────────────

class LoadMLPModel:
    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("mlp")
        return {
            "required": {
                "model_name": (models or ["(no models found)"],),
                "task":       (["omol", "omat", "oc20", "odac", "omc"],),
                "atoms":      ("ASE_ATOMS",),
            }
        }

    RETURN_TYPES = ("MLP_RESULT",)
    RETURN_NAMES = ("result",)
    FUNCTION     = "run"
    CATEGORY     = "mlp/inference"
    DESCRIPTION  = "Load a UMA/MLP model and run inference on the connected atoms."

    def run(self, model_name: str, task: str, atoms):
        from mlpui.calculator import CalculatorBuilder

        ckpt_path = folder_paths.get_full_path("mlp", model_name)
        calc  = CalculatorBuilder.from_checkpoint(ckpt_path, task=task).build()
        atoms = copy.deepcopy(atoms)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        return (MLPResult(atoms, energy, forces),)


# ─── Batch inference node ─────────────────────────────────────────────────────

class BatchResult:
    """Container for a list of MLPResult objects."""
    __slots__ = ("results",)

    def __init__(self, results):
        self.results = results   # list[MLPResult]


class BatchInference:
    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("mlp")
        return {
            "required": {
                "model_name": (models or ["(no models found)"],),
                "task":       (["omol", "omat", "oc20", "odac", "omc"],),
                "atoms_list": ("ASE_ATOMS_LIST",),
            }
        }

    RETURN_TYPES = ("BATCH_RESULT",)
    RETURN_NAMES = ("batch_result",)
    FUNCTION     = "run"
    CATEGORY     = "mlp/inference"
    DESCRIPTION  = "Run MLP inference on a list of Atoms objects (e.g. from ReadXYZBatch)."

    def run(self, model_name: str, task: str, atoms_list):
        from mlpui.calculator import CalculatorBuilder

        ckpt_path = folder_paths.get_full_path("mlp", model_name)
        calc = CalculatorBuilder.from_checkpoint(ckpt_path, task=task).build()

        results = []
        for atoms in atoms_list:
            a = copy.deepcopy(atoms)
            a.calc = calc
            energy = a.get_potential_energy()
            forces = a.get_forces()
            results.append(MLPResult(a, energy, forces))

        return (BatchResult(results),)


# ─── Output nodes ────────────────────────────────────────────────────────────

def _charge_spin(atoms):
    """Return (total_charge: int, spin_multiplicity: int) from an ASE Atoms object.
    Reads atoms.info["charge"] / atoms.info["spin"] (2S) — the same fields
    the UMAInputAdapter passes to the model.
    """
    charge = int(round(float(atoms.info.get("charge", 0))))
    spin_2s = int(round(float(atoms.info.get("spin", 0))))   # 2S stored in info
    return charge, spin_2s + 1                                # return multiplicity


class OutputEnergy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("MLP_RESULT",),
                "unit":   (list(_ENERGY_CONV.keys()),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION     = "output"
    CATEGORY     = "mlp/output"
    OUTPUT_NODE  = True
    DESCRIPTION  = "Display total energy from MLP inference results."

    def output(self, result, unit="eV"):
        factor = _ENERGY_CONV.get(unit, 1.0)
        e      = result.energy * factor
        n      = result.n_atoms
        charge, mult = _charge_spin(result.atoms)
        text = (
            f"Formula     : {result.formula}\n"
            f"N atoms     : {n}\n"
            f"Charge      : {charge:+d}\n"
            f"Multiplicity: {mult}\n"
            f"Energy      : {e:.6f} {unit}\n"
            f"E/atom      : {e / n:.6f} {unit}"
        )
        return (text,)


class OutputForces:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("MLP_RESULT",),
                "unit":   (list(_FORCE_CONV.keys()),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("forces_text",)
    FUNCTION     = "output"
    CATEGORY     = "mlp/output"
    OUTPUT_NODE  = True
    DESCRIPTION  = "Display per-atom forces from MLP inference results."

    def output(self, result, unit="eV/Å"):
        factor  = _FORCE_CONV.get(unit, 1.0)
        forces  = result.forces * factor
        symbols = result.atoms.get_chemical_symbols()
        charge, mult = _charge_spin(result.atoms)
        lines = [
            f"Forces ({unit}) — {result.formula}  [{result.n_atoms} atoms]",
            f"Charge: {charge:+d}   Multiplicity: {mult}",
            "",
        ]
        for i, (sym, f) in enumerate(zip(symbols, forces)):
            mag = float(np.linalg.norm(f))
            lines.append(
                f"  {i:3d}  {sym:2s}"
                f"  {f[0]:+9.5f}  {f[1]:+9.5f}  {f[2]:+9.5f}"
                f"  |F|={mag:.5f}"
            )
        return ("\n".join(lines),)


class OutputBatchResults:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_result": ("BATCH_RESULT",),
                "unit":         (list(_ENERGY_CONV.keys()),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION     = "output"
    CATEGORY     = "mlp/output"
    OUTPUT_NODE  = True
    DESCRIPTION  = "Display energies for all structures in a batch."

    def output(self, batch_result, unit="eV"):
        factor  = _ENERGY_CONV.get(unit, 1.0)
        results = batch_result.results
        lines   = [f"Batch results ({len(results)} structures) — Energy in {unit}", ""]
        lines.append(f"  {'#':>4}  {'Formula':<16}  {'N':>5}  {'Charge':>6}  {'Mult':>4}  {'Energy':>14}  {'E/atom':>14}")
        lines.append("  " + "-" * 74)
        for i, r in enumerate(results):
            e = r.energy * factor
            charge, mult = _charge_spin(r.atoms)
            lines.append(
                f"  {i:4d}  {r.formula:<16}  {r.n_atoms:5d}  {charge:+6d}  {mult:4d}"
                f"  {e:14.6f}  {e / r.n_atoms:14.6f}"
            )
        return ("\n".join(lines),)


# ─── Registry ─────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ReadXYZFile":            ReadXYZFile,
    "ReadXYZBatch":           ReadXYZBatch,
    "CreateAtomsFromFormula": CreateAtomsFromFormula,
    "LoadMLPModel":           LoadMLPModel,
    "BatchInference":         BatchInference,
    "OutputEnergy":           OutputEnergy,
    "OutputForces":           OutputForces,
    "OutputBatchResults":     OutputBatchResults,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReadXYZFile":            "Read XYZ File",
    "ReadXYZBatch":           "Read XYZ Batch",
    "CreateAtomsFromFormula": "Create Atoms",
    "LoadMLPModel":           "Load MLP Model",
    "BatchInference":         "Batch Inference",
    "OutputEnergy":           "Output Energy",
    "OutputForces":           "Output Forces",
    "OutputBatchResults":     "Output Batch Results",
}
