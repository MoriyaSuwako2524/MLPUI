"""Built-in nodes for MLPUI — mirrors ComfyUI's node pattern."""

import copy
import os
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


# ─── Result containers ───────────────────────────────────────────────────────

class MLPResult:
    """Carries atoms + energy + forces for one structure."""
    __slots__ = ("atoms", "energy", "forces", "formula", "n_atoms")

    def __init__(self, atoms, energy, forces):
        self.atoms   = atoms
        self.energy  = float(energy)
        self.forces  = np.asarray(forces)
        self.formula = atoms.get_chemical_formula()
        self.n_atoms = len(atoms)


class BatchResult:
    """List of MLPResult objects (single or multiple structures)."""
    __slots__ = ("results",)

    def __init__(self, results):
        self.results = results   # list[MLPResult]


# ─── Input nodes ─────────────────────────────────────────────────────────────

class ReadXYZFile:
    @classmethod
    def INPUT_TYPES(cls):
        xyz_files = folder_paths.get_xyz_list()
        placeholder = ["(upload an .xyz file first)"]
        return {
            "required": {
                "filepath": (xyz_files if xyz_files else placeholder,),
            },
            "optional": {
                "charge": ("INT", {"default": 0, "min": -20, "max": 20}),
                "spin":   ("INT", {"default": 1, "min": 1,  "max": 20}),
                "index":  ("INT", {"default": 0, "min": 0,  "max": 99999}),
                "all_frames": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("ASE_ATOMS_LIST",)
    RETURN_NAMES = ("atoms_list",)
    FUNCTION     = "read"
    CATEGORY     = "mlp/input"
    DESCRIPTION  = (
        "Read XYZ file. index selects a single frame; "
        "enable all_frames to read the full trajectory."
    )

    def read(self, filepath: str, charge: int = 0, spin: int = 1,
             index: int = 0, all_frames: bool = False):
        from ase.io import read
        full_path = folder_paths.get_xyz_path(filepath)
        raw = read(full_path, index=":" if all_frames else index)
        atoms_list = raw if isinstance(raw, list) else [raw]
        for atoms in atoms_list:
            atoms.info["charge"] = charge
            atoms.info["spin"]   = spin - 1   # 2S
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

    RETURN_TYPES = ("ASE_ATOMS_LIST",)
    RETURN_NAMES = ("atoms_list",)
    FUNCTION     = "create"
    CATEGORY     = "mlp/input"
    DESCRIPTION  = "Build an Atoms object from a chemical formula."

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
        return ([atoms],)


# ─── Inference node (single + batch) ─────────────────────────────────────────

class RunMLPModel:
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
    RETURN_NAMES = ("result",)
    FUNCTION     = "run"
    CATEGORY     = "mlp/inference"
    DESCRIPTION  = (
        "Run MLP inference on one or more structures. "
        "Accepts ASE_ATOMS_LIST from any source (ReadXYZ, DatasetBatch, etc.)."
    )

    def run(self, model_name: str, task: str, atoms_list):
        from mlpui.calculator import CalculatorBuilder

        ckpt_path = folder_paths.get_full_path("mlp", model_name)
        calc = CalculatorBuilder.from_checkpoint(ckpt_path, task=task).build()

        results = []
        for atoms in atoms_list:
            a = copy.deepcopy(atoms)
            a.calc = calc
            results.append(MLPResult(a, a.get_potential_energy(), a.get_forces()))

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
                "result": ("BATCH_RESULT",),
                "unit":   (list(_ENERGY_CONV.keys()),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION     = "output"
    CATEGORY     = "mlp/output"
    OUTPUT_NODE  = True
    DESCRIPTION  = "Display energy for one or more structures from RunMLPModel."

    def output(self, result, unit="eV"):
        factor  = _ENERGY_CONV.get(unit, 1.0)
        results = result.results

        if len(results) == 1:
            r = results[0]
            e = r.energy * factor
            charge, mult = _charge_spin(r.atoms)
            text = (
                f"Formula     : {r.formula}\n"
                f"N atoms     : {r.n_atoms}\n"
                f"Charge      : {charge:+d}\n"
                f"Multiplicity: {mult}\n"
                f"Energy      : {e:.6f} {unit}\n"
                f"E/atom      : {e / r.n_atoms:.6f} {unit}"
            )
        else:
            lines = [f"Energy ({unit})  —  {len(results)} structures", ""]
            lines.append(f"  {'#':>4}  {'Formula':<16}  {'N':>5}  {'Energy':>14}  {'E/atom':>14}")
            lines.append("  " + "─" * 60)
            for i, r in enumerate(results):
                e = r.energy * factor
                lines.append(
                    f"  {i:4d}  {r.formula:<16}  {r.n_atoms:5d}"
                    f"  {e:14.6f}  {e / r.n_atoms:14.6f}"
                )
            text = "\n".join(lines)

        return (text,)


class OutputForces:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("BATCH_RESULT",),
                "unit":   (list(_FORCE_CONV.keys()),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("forces_text",)
    FUNCTION     = "output"
    CATEGORY     = "mlp/output"
    OUTPUT_NODE  = True
    DESCRIPTION  = "Display per-atom forces. Single structure: full table; batch: per-structure summary."

    def output(self, result, unit="eV/Å"):
        factor  = _FORCE_CONV.get(unit, 1.0)
        results = result.results

        if len(results) == 1:
            r       = results[0]
            forces  = r.forces * factor
            symbols = r.atoms.get_chemical_symbols()
            charge, mult = _charge_spin(r.atoms)
            lines = [
                f"Forces ({unit}) — {r.formula}  [{r.n_atoms} atoms]",
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
        else:
            lines = [f"Force summary ({unit})  —  {len(results)} structures", ""]
            lines.append(f"  {'#':>4}  {'Formula':<16}  {'N':>5}  {'Max|F|':>10}  {'RMS|F|':>10}")
            lines.append("  " + "─" * 52)
            for i, r in enumerate(results):
                f     = r.forces * factor
                mags  = np.linalg.norm(f, axis=-1)
                lines.append(
                    f"  {i:4d}  {r.formula:<16}  {r.n_atoms:5d}"
                    f"  {mags.max():10.5f}  {np.sqrt(np.mean(mags**2)):10.5f}"
                )

        return ("\n".join(lines),)


# ─── Dataset result container ─────────────────────────────────────────────────

class DatasetResult:
    """
    Reference data for one or more frames from an NPZ dataset.

    Each frame is a dict that may contain any subset of:
      energy  : float          — total energy (eV)
      forces  : ndarray(N,3)   — atomic forces (eV/Å), first gradient level
      charges : ndarray(N,)    — partial atomic charges (e)
      dipole  : ndarray(3,)    — dipole moment vector (e·Å)
    """
    __slots__ = ("frames",)

    def __init__(self, frames: list):
        self.frames = frames   # list of dicts

    def __len__(self):
        return len(self.frames)


# ─── Dataset nodes ────────────────────────────────────────────────────────────

def _get_dataset_names():
    """Return dataset names for COMBO widgets; falls back to placeholder."""
    try:
        from mlpui.dataset import DatasetManager
        dm    = DatasetManager(folder_paths.datasets_dir)
        names = [d["name"] for d in dm.list_datasets() if "error" not in d]
        return names if names else ["(no datasets)"]
    except Exception:
        return ["(no datasets)"]


def _frame_to_dict(frame: dict) -> dict:
    """Convert a raw frame dict (from DatasetManager.get_frame) into a clean dict
    with numpy arrays for forces/charges/dipole and a plain float for energy."""
    out = {}
    e = frame.get("energy")
    if e is not None and np.isfinite(float(e)):
        out["energy"] = float(e)

    f_raw = frame.get("forces")
    if f_raw is not None:
        f = np.array(f_raw, dtype=float)
        if f.ndim == 3:         # (M, N_atoms, 3) multi-level → take first
            f = f[0]
        out["forces"] = np.where(np.isfinite(f), f, 0.0)

    q_raw = frame.get("charges")
    if q_raw is not None:
        q = np.array(q_raw, dtype=float)
        out["charges"] = np.where(np.isfinite(q), q, 0.0)

    d_raw = frame.get("dipole")
    if d_raw is not None:
        d = np.array(d_raw, dtype=float)
        if np.all(np.isfinite(d)):
            out["dipole"] = d

    return out


class DatasetLoader:
    """Load a single frame from an NPZ dataset → atoms + DATASET_RESULT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (_get_dataset_names(),),
                "index":   ("INT", {"default": 0, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("ASE_ATOMS_LIST", "DATASET_RESULT")
    RETURN_NAMES = ("atoms_list",     "ref_result")
    FUNCTION     = "load"
    CATEGORY     = "mlp/dataset"
    DESCRIPTION  = (
        "Load a single frame from an NPZ dataset. "
        "ref_result carries energy / forces / charges / dipole for comparison."
    )

    def load(self, dataset, index):
        import ase
        from mlpui.dataset import DatasetManager
        dm    = DatasetManager(folder_paths.datasets_dir)
        frame = dm.get_frame(dataset, int(index))

        pos   = np.array(frame["positions"])
        nums  = np.array(frame["numbers"], dtype=int)
        atoms = ase.Atoms(numbers=nums, positions=pos)

        return ([atoms], DatasetResult([_frame_to_dict(frame)]))


class DatasetBatch:
    """Load multiple frames from an NPZ dataset → atoms list + DATASET_RESULT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (_get_dataset_names(),),
                "start":   ("INT", {"default": 0,  "min": 0, "max": 999999}),
                "count":   ("INT", {"default": 20, "min": 1, "max": 500}),
            }
        }

    RETURN_TYPES = ("ASE_ATOMS_LIST", "DATASET_RESULT")
    RETURN_NAMES = ("atoms_list",     "ref_result")
    FUNCTION     = "load"
    CATEGORY     = "mlp/dataset"
    DESCRIPTION  = (
        "Load a batch of frames from an NPZ dataset. "
        "ref_result carries per-frame energy / forces / charges / dipole."
    )

    def load(self, dataset, start, count):
        import ase
        from mlpui.dataset.dataset_manager import _ALIASES, _resolve

        path = os.path.join(folder_paths.datasets_dir, dataset + ".npz")
        atoms_list: list  = []
        frames_data: list = []

        with np.load(path, allow_pickle=False) as npz:
            keys = list(npz.keys())
            cm   = {c: _resolve(keys, c) for c in _ALIASES if _resolve(keys, c)}

            if "energy" in cm:
                n = len(npz[cm["energy"]])
            elif "positions" in cm:
                n = npz[cm["positions"]].shape[0]
            else:
                n = npz[keys[0]].shape[0]

            end = min(int(start) + int(count), n)

            pos_arr     = npz[cm["positions"]] if "positions" in cm else None
            num_arr     = npz[cm["numbers"]]   if "numbers"   in cm else None
            energy_arr  = npz[cm["energy"]]    if "energy"    in cm else None
            forces_arr  = npz[cm["forces"]]    if "forces"    in cm else None
            charges_arr = npz[cm["charges"]]   if "charges"   in cm else None
            dipole_arr  = npz[cm["dipole"]]    if "dipole"    in cm else None

            for i in range(int(start), end):
                # Build ASE Atoms
                if pos_arr is not None and num_arr is not None:
                    pos  = pos_arr[i]
                    nums = num_arr[i] if num_arr.ndim == 2 else num_arr
                    atoms_list.append(ase.Atoms(numbers=nums.astype(int), positions=pos))

                fd: dict = {}

                if energy_arr is not None:
                    e = float(energy_arr[i])
                    if np.isfinite(e):
                        fd["energy"] = e

                if forces_arr is not None:
                    f = forces_arr[i]
                    if f.ndim == 3:
                        f = f[0]
                    fd["forces"] = np.where(np.isfinite(f), f, 0.0)

                if charges_arr is not None:
                    q = charges_arr[i]
                    fd["charges"] = np.where(np.isfinite(q), q, 0.0)

                if dipole_arr is not None:
                    d = dipole_arr[i]
                    if np.all(np.isfinite(d)):
                        fd["dipole"] = d

                frames_data.append(fd)

        return (atoms_list, DatasetResult(frames_data))


# ─── Evaluate node ────────────────────────────────────────────────────────────

# Mapping: field name → (label, unit, extractor from MLPResult or None)
_EVAL_FIELDS = {
    "energy":  ("Energy",    "eV",   lambda r: r.energy),
    "forces":  ("Force RMS", "eV/Å", lambda r: float(np.sqrt(np.mean(np.sum(r.forces**2, axis=-1)))) if r.forces is not None else None),
    "charges": ("Charges",   "e",    None),   # MLP doesn't yet output charges
    "dipole":  ("Dipole",    "e·Å",  None),   # MLP doesn't yet output dipole
}


class EvalParity:
    """
    Compare reference (DATASET_RESULT) with predicted (BATCH_RESULT).

    Automatically detects which fields exist in both reference and prediction,
    computes MAE / RMSE / R² for each, and generates a multi-panel parity plot.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref":  ("DATASET_RESULT",),
                "pred": ("BATCH_RESULT",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("metrics", "parity_plots")
    FUNCTION     = "evaluate"
    CATEGORY     = "mlp/evaluate"
    OUTPUT_NODE  = True
    DESCRIPTION  = (
        "Compare dataset reference data with BatchInference predictions. "
        "Auto-matches available fields (energy, forces, …) and plots each."
    )

    def evaluate(self, ref: DatasetResult, pred: BatchResult):
        n = min(len(ref.frames), len(pred.results))
        if n == 0:
            return ("No frames to compare.", None)

        # Fields present in at least one reference frame
        ref_fields = set()
        for fd in ref.frames[:n]:
            ref_fields |= fd.keys()

        plots: list = []   # (ref_arr, pred_arr, label, unit)
        metrics_parts: list = []

        for field, (label, unit, pred_extractor) in _EVAL_FIELDS.items():
            if field not in ref_fields:
                continue
            if pred_extractor is None:
                continue   # prediction doesn't support this field yet

            ref_vals, pred_vals = [], []
            for i in range(n):
                fd  = ref.frames[i]
                res = pred.results[i]

                # Reference value
                rv = fd.get(field)
                if rv is None:
                    continue
                if field == "energy":
                    rv_scalar = float(rv)
                elif field == "forces":
                    rv_scalar = float(np.sqrt(np.mean(np.sum(rv**2, axis=-1))))
                elif field == "charges":
                    rv_scalar = float(np.sum(rv))
                elif field == "dipole":
                    rv_scalar = float(np.linalg.norm(rv))
                else:
                    continue

                # Predicted value
                pv_scalar = pred_extractor(res)
                if pv_scalar is None or not np.isfinite(pv_scalar):
                    continue
                if not np.isfinite(rv_scalar):
                    continue

                ref_vals.append(rv_scalar)
                pred_vals.append(pv_scalar)

            if len(ref_vals) < 2:
                continue

            ref_a  = np.array(ref_vals)
            pred_a = np.array(pred_vals)
            mae    = float(np.mean(np.abs(ref_a - pred_a)))
            rmse   = float(np.sqrt(np.mean((ref_a - pred_a) ** 2)))
            max_e  = float(np.max(np.abs(ref_a - pred_a)))
            r2     = float(np.corrcoef(ref_a, pred_a)[0, 1] ** 2) \
                     if ref_a.std() > 0 and pred_a.std() > 0 else float("nan")

            metrics_parts.append(
                f"{label}  (n={len(ref_a)})\n"
                f"{'─'*34}\n"
                f"  MAE     = {mae:.6f}  {unit}\n"
                f"  RMSE    = {rmse:.6f}  {unit}\n"
                f"  Max err = {max_e:.6f}  {unit}\n"
                f"  R²      = {r2:.6f}\n"
            )
            plots.append((ref_a, pred_a, label, unit, mae, rmse))

        if not plots:
            return ("No common fields found between reference and prediction.\n"
                    "(Currently supported: energy, forces)", None)

        metrics_str = "\n".join(metrics_parts)
        img = self._make_plots(plots)
        return (metrics_str, img)

    def _make_plots(self, plots):
        try:
            import io, base64
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_p = len(plots)
            fig, axes = plt.subplots(1, n_p, figsize=(5 * n_p, 5), dpi=110,
                                     squeeze=False)
            fig.patch.set_facecolor("#1e1e1e")

            for ax, (ref, pred, label, unit, mae, rmse) in zip(axes[0], plots):
                ax.set_facecolor("#282828")
                ax.scatter(ref, pred, s=14, alpha=0.7,
                           color="#4a9eff", edgecolors="none", zorder=3)

                lo  = min(ref.min(), pred.min())
                hi  = max(ref.max(), pred.max())
                pad = (hi - lo) * 0.06 or 0.1
                lim = [lo - pad, hi + pad]
                ax.plot(lim, lim, "--", color="#f44336", lw=1.2, alpha=0.8, zorder=2)
                ax.set_xlim(lim); ax.set_ylim(lim)
                ax.set_aspect("equal")

                ax.set_xlabel(f"Reference ({unit})", color="#cccccc", fontsize=9)
                ax.set_ylabel(f"Predicted ({unit})", color="#cccccc", fontsize=9)
                ax.set_title(
                    f"{label}\nMAE={mae:.4f}  RMSE={rmse:.4f} {unit}",
                    color="#cccccc", fontsize=8,
                )
                ax.tick_params(colors="#888888", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#3a3a3a")
                ax.grid(True, color="#333333", linewidth=0.5, zorder=1)

            fig.tight_layout(pad=1.2)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
        except ImportError:
            return None


# ─── Registry ─────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ReadXYZFile":            ReadXYZFile,
    "CreateAtomsFromFormula": CreateAtomsFromFormula,
    "RunMLPModel":            RunMLPModel,
    "OutputEnergy":           OutputEnergy,
    "OutputForces":           OutputForces,
    "DatasetLoader":          DatasetLoader,
    "DatasetBatch":           DatasetBatch,
    "EvalParity":             EvalParity,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReadXYZFile":            "Read XYZ File",
    "CreateAtomsFromFormula": "Create Atoms",
    "RunMLPModel":            "Run MLP Model",
    "OutputEnergy":           "Output Energy",
    "OutputForces":           "Output Forces",
    "DatasetLoader":          "Dataset Loader",
    "DatasetBatch":           "Dataset Batch",
    "EvalParity":             "Eval Parity",
}
