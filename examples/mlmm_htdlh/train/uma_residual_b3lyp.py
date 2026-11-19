
"""UMA-s-1p2 (omol) gas predictions for the htdlh stage-1 frames, then residual v2 = QMMM - UMA - E_charge - offset."""
import os, sys, json, time, hashlib, numpy as np, torch
from pathlib import Path
from ase import Atoms
from fairchem.core import pretrained_mlip, FAIRChemCalculator
CK = '/home/moriya2524/mlp/sources/UMA/UMA/checkpoints/uma-s-1p2.pt'
S1 = Path('/scratch/moriya2524/qeq_stage1_b3lyp_qc63_v1'); R1 = Path('/scratch/moriya2524/htdlh_qeq_residual_b3lyp_qc63_s42')
R2 = Path('/scratch/moriya2524/htdlh_qeq_uma_residual_b3lyp_qc63_s42'); U = Path('/home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/uma_s1p2_gas')
assert not R2.exists(); U.mkdir(parents=True, exist_ok=True)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
t0 = time.time()
pu = pretrained_mlip.load_predict_unit(CK, device='cuda')
calc = FAIRChemCalculator(pu, task_name='omol')
print('loaded', round(time.time() - t0, 1), 's', flush=True)
roles = ('train', 'validation', 'test', 'guard'); uma = {}
for role in roles:
    z = np.load(S1 / role / 'z.npy'); pos = np.load(S1 / role / 'pos.npy'); Q = np.load(S1 / role / 'charge.npy')
    E = np.zeros(len(pos)); F = np.zeros(pos.shape)
    for i in range(len(pos)):
        a = Atoms(numbers=z, positions=pos[i]); a.info.update(charge=int(round(float(Q[i]))), spin=1); a.calc = calc
        E[i] = a.get_potential_energy(); F[i] = a.get_forces()
        if i % 2000 == 0: print(role, i, round(time.time() - t0, 1), flush=True)
    np.save(U / f'{role}_energy.npy', E); np.save(U / f'{role}_forces.npy', F); uma[role] = (E, F)
    print(role, 'done', len(pos), round(time.time() - t0, 1), flush=True)
gE = {r: np.load(S1 / r / 'extra/gas_energy.npy') for r in roles}; gF = {r: np.load(S1 / r / 'extra/gas_forces.npy') for r in roles}
offset = float(np.mean(gE['train'] - uma['train'][0]))
stats = {r: dict(gas_minus_uma_energy_std=float(np.std(gE[r] - uma[r][0])), gas_minus_uma_energy_rmse_after_offset=float(np.sqrt(np.mean((gE[r] - uma[r][0] - offset) ** 2))),
                 gas_minus_uma_force_rms=float(np.sqrt(np.mean((gF[r] - uma[r][1]) ** 2)))) for r in roles}
print('offset', offset, json.dumps(stats), flush=True)
for role in roles:
    src, dst = R1 / role, R2 / role; (dst / 'extra').mkdir(parents=True)
    for p in src.glob('*.npy'):
        if p.name in ('energy.npy', 'forces.npy'): continue
        os.symlink(p.resolve(), dst / p.name)
    for p in (src / 'extra').glob('*.npy'): os.symlink(p.resolve(), dst / 'extra' / p.name)
    e1, f1 = np.load(src / 'energy.npy'), np.load(src / 'forces.npy')
    np.save(dst / 'energy.npy', e1 + gE[role] - uma[role][0] - offset)
    np.save(dst / 'forces.npy', f1 + gF[role] - uma[role][1])
    np.save(dst / 'extra/uma_energy.npy', uma[role][0]); np.save(dst / 'extra/uma_forces.npy', uma[role][1])
    e2, f2 = np.load(dst / 'energy.npy'), np.load(dst / 'forces.npy')
    stats[role].update(residual_v2_energy_rms=float(np.sqrt(np.mean(e2 ** 2))), residual_v2_force_rms=float(np.sqrt(np.mean(f2 ** 2))),
                       residual_v1_energy_rms=float(np.sqrt(np.mean(e1 ** 2))), residual_v1_force_rms=float(np.sqrt(np.mean(f1 ** 2))))
man = dict(definition='energy/forces = QMMM label - E_UMA(uma-s-1p2, omol, charge -1 (per-frame charge.npy), spin 1) - E_charge(QEq best.pt) - energy_offset; mm_potential(_gradient) as residual v1 (UMA has no MM dependence)',
           energy_offset_eV=offset, offset_definition='mean over train of (E_gas B3LYP-D3(BJ)/6-31G* - E_UMA)', uma_checkpoint=CK, uma_checkpoint_sha256=sha(CK),
           qeq_residual_source=str(R1), qeq_residual_manifest=json.loads((R1 / 'train' / 'residual_manifest.json').read_text())['checkpoint_sha256'], stats=stats)
(R2 / 'manifest.json').write_text(json.dumps(man, indent=1)); (U / 'manifest.json').write_text(json.dumps(man, indent=1))
print('BUILD DONE', round(time.time() - t0, 1), json.dumps(stats))
