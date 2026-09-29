"""Native UMA contract tests and opt-in real-checkpoint integration checks."""
import os

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule

from mlpui.calculator import CalculatorBuilder
from mlpui.models.uma.predictor import DatasetEnergyHead


def test_dataset_head_routing():
    class Shape:
        sphere_channels = hidden_channels = 2
    head = DatasetEnergyHead(Shape(), {'omol': 'omol', 'omat': 'omat'})
    with torch.no_grad():
        for layer in head.head.energy_block:
            if hasattr(layer, 'weights'):
                layer.weights.zero_()
                layer.bias.zero_()
        head.head.energy_block[0].bias.fill_(1)
        head.head.energy_block[2].weights[1].fill_(1)
        head.head.energy_block[4].weights[1].fill_(1)
    emb = torch.zeros(2, 1, 2)
    data = {'natoms': torch.tensor([2]), 'batch': torch.zeros(2, dtype=torch.long), 'dataset': ['omat']}
    assert head(emb, data).item() == 0
    data['dataset'] = ['omol']
    assert head(emb, data).item() > 0


@pytest.fixture(scope='module')
def real_builder():
    checkpoint = os.environ.get('MLPUI_UMA_CHECKPOINT')
    if not checkpoint:
        pytest.skip('Set MLPUI_UMA_CHECKPOINT to run actual UMA integration tests')
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return CalculatorBuilder.from_checkpoint(checkpoint, task='omol', spin=1,
        device=device, dtype=torch.float64)


@pytest.mark.integration
def test_real_forces_finite_difference_and_translation(real_builder):
    atoms = molecule('H2O')
    atoms.positions[1] += [0.07, 0.03, -0.02]
    atoms.calc = real_builder.build()
    energy = atoms.get_potential_energy()
    force = atoms.get_forces()
    h = 1e-4
    fd = np.zeros_like(force)
    for i in range(len(atoms)):
        for j in range(3):
            atoms.positions[i, j] += h
            plus = atoms.get_potential_energy()
            atoms.positions[i, j] -= 2 * h
            minus = atoms.get_potential_energy()
            atoms.positions[i, j] += h
            fd[i, j] = -(plus - minus) / (2 * h)
    np.testing.assert_allclose(force, fd, atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-8)
    atoms.translate([1.1, -2.2, 0.3])
    assert atoms.get_potential_energy() == pytest.approx(energy, abs=1e-7)
    np.testing.assert_allclose(atoms.get_forces(), force, atol=1e-7)
    print('Maximum finite-difference force error:', np.abs(force - fd).max())


@pytest.mark.integration
def test_native_calculator_validation(real_builder):
    from dataclasses import replace
    with pytest.raises(ValueError, match='spin'):
        replace(real_builder, spin=0).build()
    with pytest.raises(ValueError, match='charge'):
        replace(real_builder, charge=0.5).build()
    with pytest.raises(ValueError, match='omol'):
        replace(real_builder, task='omat').build()
    atoms = molecule('H2O')
    atoms.pbc = True
    atoms.calc = real_builder.build()
    with pytest.raises(ValueError, match='nonperiodic'):
        atoms.get_potential_energy()
    atom = Atoms('H')
    atom.calc = real_builder.build()
    with pytest.raises(ValueError, match='reference'):
        atom.get_potential_energy()
    atom.calc = replace(real_builder, spin=2, atom_refs={'omol': {1: {0: -13.4}}}).build()
    with pytest.warns(UserWarning, match='Single-atom'):
        assert atom.get_potential_energy() == -13.4
    np.testing.assert_array_equal(atom.get_forces(), np.zeros((1, 3)))


@pytest.mark.integration
def test_checkpoint_rejects_missing_head_weights(real_builder):
    model = real_builder.model_patcher.model
    state = dict(model.state_dict())
    key = next(k for k in state if k.startswith('output_heads.'))
    del state[key]
    with pytest.raises(RuntimeError, match='Missing key'):
        model.load_state_dict(state, strict=True)


@pytest.mark.integration
def test_real_short_nvt(real_builder, tmp_path):
    from scripts.md.nvt import parse_args, run
    from ase.io import write, read
    import json
    xyz = tmp_path / 'water.xyz'
    write(xyz, molecule('H2O'))
    args = parse_args(['--xyz', str(xyz), '--checkpoint', os.environ['MLPUI_UMA_CHECKPOINT'],
        '--steps', '20', '--timestep-fs', '0.1', '--save-every', '3', '--log-every', '3',
        '--output', str(tmp_path / 'md')])
    run(args, lambda _: real_builder.build())
    frames = read(str(args.output / 'trajectory.traj'), index=':')
    assert [a.info['md_step'] for a in frames] == [0, 3, 6, 9, 12, 15, 18, 20]
    assert all(np.isfinite(a.get_potential_energy()) for a in frames)
    assert all(np.isfinite(a.get_forces()).all() for a in frames)
    assert json.loads((args.output / 'run.json').read_text())['status'] == 'completed'
