"""Join verified ESP labels onto existing residual E/F train/validation datasets.

Requires a reviewed baseline contract from the original E/F preparation.
Never estimates it from the test set, changes E/F values, or writes source data.
Usage: python -m scripts.prepare_cm_fourtarget --help
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from mlpui.data import NpyDataset
from mlpui.qmmm_targets import fixed_charge_mm_properties, train_target_scales


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare(args):
    source=Path(args.source);esp=Path(args.esp);gas=Path(args.gas)
    contract=json.loads(Path(args.contract).read_text())
    required={'gas_charge_checkpoint_sha256','gas_prediction_sha256','coulomb_constant','charge_projection',
              'qm_force_includes_dq_dR','mm_self_excluded','ef_source_sha256'}
    if not required<=contract.keys() or contract['qm_force_includes_dq_dR'] is not True or contract['mm_self_excluded'] is not True:
        raise ValueError('Reviewed original E/F baseline contract is required')
    if contract['charge_projection'] not in ('as_saved','uniform_total_charge'):
        raise ValueError('Unknown charge projection; do not guess')
    metadata=json.loads((gas/'metadata.json').read_text())
    if metadata['provenance']['checkpoint']['sha256']!=contract['gas_charge_checkpoint_sha256']:
        raise ValueError('Gas checkpoint identity mismatch')
    if sha(gas/'charges_pred.npy')!=contract['gas_prediction_sha256']:
        raise ValueError('Gas prediction identity mismatch')
    verification=json.loads((esp.parent/'independent_verification.json').read_text())
    if verification['status']!='COMPLETE_VERIFIED' or verification['n_frames']!=16000:
        raise ValueError('Requires verified complete ESP export')
    export=json.loads((esp/'metadata.json').read_text())
    for name,record in export['files'].items():
        if sha(esp/name)!=record['sha256']:raise ValueError('ESP checksum mismatch: '+name)
    load=lambda folder,name:np.load(folder/(name+'.npy'),allow_pickle=False,mmap_mode='r')
    ids=load(esp,'source_index');offsets=load(esp,'mm_offsets')
    if not np.array_equal(ids,np.arange(16000)) or not np.array_equal(offsets,load(source,'mm_offsets')):
        raise ValueError('Source order/offset mismatch')
    if not np.array_equal(load(gas,'frame_index'),ids):raise ValueError('Gas frame IDs mismatch')
    masks={role:load(gas,'is_'+role) for role in ('train','validation','test','guard')}
    if any(v.dtype.kind!='b' or v.shape!=(16000,) for v in masks.values()):raise ValueError('Invalid split masks')
    if not np.all(sum(v.astype(int) for v in masks.values())==1):raise ValueError('Splits overlap or omit frames')
    if [int(v.sum()) for v in masks.values()]!=[11200,1600,1600,1600]:raise ValueError('Unexpected fixed split sizes')
    q=load(gas,'charges_pred');pos=load(source,'full_qm_coords');z=load(source,'full_qm_type')
    if q.shape!=pos.shape[:2] or not np.isfinite(q).all():raise ValueError('Invalid gas charges')
    mm=load(source,'mm_coords_concat');mq=load(source,'mm_charges_concat')
    phi=load(esp,'ESP_MM_eV_per_e');g=load(esp,'G_ESP_MM_eV_per_e_A')
    output=Path(args.output)
    if output.exists():raise FileExistsError('Output must be new; refusing overwrite')
    datasets={}
    # Validate identity and original E/F hashes before creating any output.
    for role,path in [('train',args.train),('validation',args.validation)]:
        d=NpyDataset(path);expected=np.flatnonzero(masks[role])
        if any(k.startswith('baseline_') for k in d.arrays):raise ValueError('Requires residual-only E/F data')
        if set(d.targets)&d.arrays.keys()!={'energy','forces'}:raise ValueError('Input must contain only original E/F targets')
        if not np.array_equal(d.arrays.get('source_index'),expected):raise ValueError(role+' source IDs differ from fixed split')
        for key in ('energy','forces'):
            if sha(Path(path)/(key+'.npy'))!=contract['ef_source_sha256'][role][key]:raise ValueError(role+' E/F identity mismatch')
        for row,i in enumerate(expected):
            s=d[row];a,b=map(int,offsets[i:i+2])
            for actual,wanted in [(s['z'],z if z.ndim==1 else z[i]),(s['pos'],pos[i]),(s['mm_pos'],mm[a:b]),(s['mm_charges'],mq[a:b])]:
                if not np.array_equal(actual,wanted):raise ValueError(role+' geometry/MM charges mismatch')
        datasets[role]=d
    output.mkdir(parents=True)
    for role,d in datasets.items():
        dest=output/role;dest.mkdir()
        for name in d.arrays:shutil.copy2(Path(d.arrays[name].filename),dest/(name+'.npy'))
        n=int(d.arrays['mm_offsets'][-1]);pot=np.lib.format.open_memmap(dest/'mm_potential.npy',mode='w+',dtype='float64',shape=(n,))
        grad=np.lib.format.open_memmap(dest/'mm_potential_gradient.npy',mode='w+',dtype='float64',shape=(n,3))
        for row,i in enumerate(d.arrays['source_index']):
            s=d[row];a,b=map(int,offsets[i:i+2]);c,e=map(int,d.arrays['mm_offsets'][row:row+2]);charges=np.array(q[i])
            if contract['charge_projection']=='uniform_total_charge':charges=charges-charges.mean()+float(s['charge'])/len(charges)
            bp,bg=fixed_charge_mm_properties(s['pos'],s['mm_pos'],charges,coulomb_constant=contract['coulomb_constant'])
            pot[c:e]=phi[a:b]-bp;grad[c:e]=g[a:b]-bg
        pot.flush();grad.flush();del pot,grad
    scales=train_target_scales(NpyDataset(output/'train'))
    (output/'scales.json').write_text(json.dumps(scales,indent=2))
    report=dict(status='PREPARED',baseline_contract=contract,contract_sha256=sha(args.contract),
                gas_prediction_sha256=sha(gas/'charges_pred.npy'),esp_verification=verification,
                splits={k:int(v.sum()) for k,v in masks.items()},test_attached=False,
                target_semantics='residual E/F unchanged; gas fixed-charge Coulomb subtracted from ESP/GESP',scales=scales)
    (output/'preparation.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','esp','gas','contract','train','validation','output'):parser.add_argument('--'+name,required=True)
    print(json.dumps(prepare(parser.parse_args()),indent=2))
