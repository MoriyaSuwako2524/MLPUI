
import json, sys, time, numpy as np
from pathlib import Path
from mlpui.potentials import QMMMInput
from mlpui.potentials.spec import load_spec, build_potential
B = Path('/scratch/moriya2524/htdlh_model_bundle_b3lyp_v1'); R = '/scratch/moriya2524/htdlh_qeq_uma_residual_b3lyp_qc63_s42/test'
L = lambda n: np.load(f'{R}/{n}.npy', allow_pickle=False)
off = json.load(open(B/'models/uma_residual_manifest.json'))['energy_offset_eV']
z, pos, mo, mp, mq = L('z'), L('pos'), L('mm_offsets'), L('mm_pos'), L('mm_charges')
S = len(mo) - 1; N = pos.reshape(S, -1, 3).shape[1]; pos = pos.reshape(S, N, 3)
z = z.reshape(S, N) if z.size == S*N else np.broadcast_to(z.reshape(-1), (S, N))
uE, uF = np.load(f'{R}/extra/uma_energy.npy').ravel(), np.load(f'{R}/extra/uma_forces.npy').reshape(S, N, 3)
refE = L('energy').ravel() + uE + L('qeq_energy').ravel() + off
refF = L('forces').reshape(S, N, 3) + uF + L('qeq_forces').reshape(S, N, 3)
rphi = L('mm_potential').ravel() + L('qeq_mm_potential').ravel()
rG = L('mm_potential_gradient').reshape(-1, 3) + L('qeq_mm_potential_gradient').reshape(-1, 3)
idx = np.linspace(0, S - 1, 8).astype(int)
out = {}
for tag, specname in (('fp32', 'htdlh_full_spec.json'), ('fp64', 'htdlh_full_spec_fp64.json')):
    pot = build_potential(load_spec(B / specname))
    rec = {k: [] for k in ('E', 'F', 'phi', 'G', 't')}
    for i in idx:
        a, b = mo[i], mo[i + 1]
        inp = QMMMInput(z[i].astype(np.int64), pos[i].astype(np.float64), -1, 1, mp[a:b].astype(np.float64), mq[a:b].astype(np.float64))
        t0 = time.perf_counter(); o = pot.evaluate(inp, ['energy', 'forces', 'mm_potential', 'mm_potential_gradient']); rec['t'].append(time.perf_counter() - t0)
        rec['E'].append(o['energy']); rec['F'].append(o['forces']); rec['phi'].append(o['mm_potential']); rec['G'].append(o['mm_potential_gradient'])
    out[tag] = rec
    eE = np.array(rec['E']) - refE[idx]; eF = np.concatenate([f - refF[i] for f, i in zip(rec['F'], idx)])
    ephi = np.concatenate([p - rphi[mo[i]:mo[i+1]] for p, i in zip(rec['phi'], idx)])
    eG = np.concatenate([g - rG[mo[i]:mo[i+1]] for g, i in zip(rec['G'], idx)])
    k = 23.0605
    print(tag, 'RMSE kcal: E %.3f F %.3f phi %.3f GESP %.3f' % tuple(k*np.sqrt(np.mean(x**2)) for x in (eE, eF, ephi, eG)), 'median s/frame %.3f' % np.median(rec['t'][1:]))
d = {'frame_index_in_test': idx, 'z': z[idx], 'pos': pos[idx], 'charge': -1, 'spin': 1,
     'mm_offsets': np.concatenate([[0], np.cumsum([mo[i+1]-mo[i] for i in idx])]),
     'mm_pos': np.concatenate([mp[mo[i]:mo[i+1]] for i in idx]), 'mm_charges': np.concatenate([mq[mo[i]:mo[i+1]] for i in idx]),
     'ref_qmmm_energy_eV': refE[idx], 'ref_qmmm_forces': np.stack([refF[i] for i in idx]),
     'ref_mm_potential': np.concatenate([rphi[mo[i]:mo[i+1]] for i in idx]), 'ref_mm_potential_gradient': np.concatenate([rG[mo[i]:mo[i+1]] for i in idx])}
for tag in out:
    d[f'model_{tag}_energy_eV'] = np.array(out[tag]['E']); d[f'model_{tag}_forces'] = np.stack(out[tag]['F'])
    d[f'model_{tag}_mm_potential'] = np.concatenate(out[tag]['phi']); d[f'model_{tag}_mm_potential_gradient'] = np.concatenate(out[tag]['G'])
(B / 'validation').mkdir(exist_ok=True); np.savez(B / 'validation/test8_reference.npz', **d)
print('VALIDATION_DONE', idx.tolist())
