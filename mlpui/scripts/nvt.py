"""Langevin NVT using MLPUI's CalculatorBuilder (energies eV, lengths Angstrom).

Example: python -m mlpui.scripts.nvt --xyz water.xyz --checkpoint uma.pt
         --temperature-k 300 --duration-ps 1 --timestep-fs 0.5 --output run01

The native UMA backend supports omol, nonperiodic molecules, energy and forces.
This runner checks both properties before integrating.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xyz", type=Path, required=True, help="XYZ/extxyz; coordinates in Angstrom")
    parser.add_argument("--frame", type=int, default=0, help="Input frame index (default: 0)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--atom-refs", type=Path, help="Isolated atom references YAML (only needed for single atoms)")
    parser.add_argument("--temperature-k", type=positive_float, default=300.0)
    parser.add_argument("--timestep-fs", type=positive_float, default=0.5)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument("--duration-ps", type=positive_float)
    duration.add_argument("--steps", type=positive_int)
    parser.add_argument("--friction-per-ps", type=positive_float, default=10.0,
                        help="Langevin friction in ps^-1 (default: 10; relaxation time 100 fs)")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=positive_int, default=1,
                        help="UMA spin multiplicity, 2S+1 (default: 1)")
    parser.add_argument("--task", default="omol")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=positive_int, default=10)
    parser.add_argument("--log-every", type=positive_int, default=10)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--periodic", action="store_true",
                        help="Use full PBC; requires a nonsingular cell in extxyz")
    parser.add_argument("--fix-com", action="store_true", help="Constrain center of mass with ASE FixCom")
    parser.add_argument("--check-only", action="store_true", help="Validate energy/forces without MD")
    args = parser.parse_args(argv)
    for name in ("xyz", "checkpoint"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f"{name} file does not exist: {path}")
        setattr(args, name, path)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.duration_ps is not None:
        count = args.duration_ps * 1000 / args.timestep_fs
        if not math.isfinite(count) or not math.isclose(count, round(count), rel_tol=0, abs_tol=1e-7):
            parser.error("duration must be an integer multiple of timestep; alternatively use --steps")
        args.steps = round(count)
        if args.steps < 1:
            parser.error("duration must be at least one timestep")
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    return args


def build_calculator(args):
    from mlpui.calculator import CalculatorBuilder

    return CalculatorBuilder.from_checkpoint(
        str(args.checkpoint), task=args.task, charge=args.charge, spin=args.spin,
        device=args.device, properties=["energy", "forces"], keep_on_device=True,
        atom_refs=str(args.atom_refs) if args.atom_refs else None,
    ).build()


def run(args, calculator_factory=None):
    """calculator_factory is an injection point for testing, not a CLI fallback."""
    import numpy as np
    from ase import units
    from ase.constraints import FixCom
    from ase.io import read, write
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = read(str(args.xyz), index=args.frame, format="extxyz")
    if len(atoms) == 0 or not np.isfinite(atoms.positions).all():
        raise ValueError("Input must contain atoms with finite positions")
    if not np.isfinite(atoms.cell.array).all():
        raise ValueError("Input cell must be finite")
    if atoms.constraints:
        raise ValueError("Input constraints are unsupported; use an unconstrained XYZ")
    if args.periodic and abs(np.linalg.det(atoms.cell.array)) < 1e-12:
        raise ValueError("--periodic requires a nonsingular cell in extxyz")
    atoms.set_pbc(args.periodic)
    if args.fix_com:
        if len(atoms) < 2:
            raise ValueError("--fix-com requires at least two atoms")
        atoms.set_constraint(FixCom())
    atoms.info.update(charge=args.charge, spin=args.spin)
    atoms.calc = (calculator_factory or build_calculator)(args)

    def evaluate():
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces())
        if not math.isfinite(energy) or forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError("Calculator returned invalid energy or forces")
        return energy, float(np.linalg.norm(forces, axis=1).max())

    try:
        energy, fmax = evaluate()
    except Exception as exc:
        raise RuntimeError(
            "MLPUI energy/force preflight failed; no MD was started. "
            "Check checkpoint compatibility, task, inputs and the installed dependencies. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    print(f"Preflight: {len(atoms)} atoms, E={energy:.8f} eV, max |F|={fmax:.6f} eV/Angstrom", flush=True)
    if args.check_only:
        return

    # Thermalize all Cartesian degrees of freedom; optionally remove only COM.
    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_k, rng=rng)
    if args.fix_com:
        atoms.set_momenta(atoms.get_momenta())  # Apply FixCom to initial momenta.
    dyn = Langevin(atoms, timestep=args.timestep_fs * units.fs,
                   temperature_K=args.temperature_k,
                   friction=args.friction_per_ps / (1000 * units.fs),
                   fixcm=False, rng=rng)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(status="running", completed_steps=0,
                    actual_duration_ps=args.steps * args.timestep_fs / 1000,
                    initial_energy_ev=energy, initial_max_force_ev_per_angstrom=fmax)
    metadata_path = args.output / "run.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    save_metadata()
    started = time.monotonic()
    try:
        with Trajectory(str(args.output / "trajectory.traj"), "w", atoms,
                        properties=["energy", "forces"]) as trajectory, \
                (args.output / "thermo.csv").open("w", newline="", encoding="utf-8") as log:
            writer = csv.writer(log)
            writer.writerow(["step", "time_ps", "temperature_K", "potential_eV", "kinetic_eV",
                             "total_eV", "max_force_eV_per_A", "elapsed_s"])
            last_saved = last_logged = -1

            def record(force=False):
                nonlocal last_saved, last_logged
                epot, max_force = evaluate()
                if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.get_momenta()).all():
                    raise ValueError("Non-finite positions or momenta during MD")
                step = dyn.nsteps
                atoms.info.update(md_step=step, time_ps=step * args.timestep_fs / 1000)
                if step != last_saved and (force or step % args.save_every == 0):
                    trajectory.write(atoms)
                    last_saved = step
                if step != last_logged and (force or step % args.log_every == 0):
                    ekin = float(atoms.get_kinetic_energy())
                    temp = float(atoms.get_temperature())
                    writer.writerow([step, atoms.info["time_ps"], temp, epot, ekin,
                                     epot + ekin, max_force, time.monotonic() - started])
                    log.flush()
                    print(f"step={step}/{args.steps} T={temp:.2f} K Epot={epot:.8f} eV", flush=True)
                    last_logged = step
                metadata["completed_steps"] = step

            dyn.attach(record, interval=1)
            dyn.run(args.steps)
            record(force=True)  # Include final frame even when intervals do not divide steps.
            write(str(args.output / "final.extxyz"), atoms, format="extxyz")
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["elapsed_s"] = time.monotonic() - started
        save_metadata()
    print(f"Completed {args.steps} steps; outputs: {args.output}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted; saved trajectory and run.json retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
