"""Export native/reference UMA predictions for comparison in separate environments.

FAIRChem is imported only with --backend fairchem and is a validation dependency.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule


def structures():
    water = molecule('H2O')
    stretched = water.copy()
    stretched.positions[1] += [0.07, 0.03, -0.02]
    rotated = water.copy()
    rotated.rotate(37, [1, 2, 3])
    rotated.translate([1.4, -0.8, 2.3])
    return [('water', water, 0, 1), ('stretched_water', stretched, 0, 1),
            ('rotated_water', rotated, 0, 1), ('methane', molecule('CH4'), 0, 1),
            ('ammonia', molecule('NH3'), 0, 1),
            ('hydroxide', Atoms('OH', positions=[[0, 0, 0], [0, 0, 0.97]]), -1, 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['native', 'fairchem'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--atom-refs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    if args.backend == 'native':
        from mlpui.calculator import CalculatorBuilder
        builder = CalculatorBuilder.from_checkpoint(args.checkpoint, device=args.device)
    else:
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        import yaml
        refs = yaml.safe_load(Path(args.atom_refs).read_text()) if args.atom_refs else None
        predictor = load_predict_unit(args.checkpoint, device=args.device, atom_refs=refs,
            inference_settings=InferenceSettings(merge_mole=False, compile=False,
                activation_checkpointing=False, use_quaternion_wigner=True, tf32=False), workers=1)
    records = []
    for name, atoms, charge, spin in structures():
        atoms.info.update(charge=charge, spin=spin)
        if args.backend == 'native':
            builder.charge, builder.spin = charge, spin
            calc = builder.build()
        else:
            calc = FAIRChemCalculator(predictor, task_name='omol')
        atoms.calc = calc
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        assert np.isfinite(energy) and np.isfinite(forces).all()
        records.append(dict(name=name, energy=float(energy), forces=forces.tolist()))
        print(name, energy, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
