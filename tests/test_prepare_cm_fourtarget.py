import argparse
import json
import numpy as np
import pytest
from scripts.prepare_cm_fourtarget import prepare,sha


def test_preparation_preserves_ef_split_and_train_only_weights(tmp_path):
    n=16000
    source=tmp_path/'source';esp=tmp_path/'esp'/'npys';gas=tmp_path/'gas'
    for p in (source,esp,gas):p.mkdir(parents=True)
    save=lambda p,k,v:np.save(p/(k+'.npy'),v)
    pos=np.zeros((n,1,3));mm=np.tile([2.,0.,0.],(n,1));mq=np.ones(n)*.2
    for k,v in dict(full_qm_type=np.array([1]),full_qm_coords=pos,mm_coords_concat=mm,mm_charges_concat=mq,mm_offsets=np.arange(n+1)).items():save(source,k,v)
    for k,v in dict(source_index=np.arange(n),mm_offsets=np.arange(n+1),ESP_MM_eV_per_e=np.ones(n)*3,G_ESP_MM_eV_per_e_A=np.ones((n,3))*.4).items():save(esp,k,v)
    (esp/'metadata.json').write_text(json.dumps({'files':{p.name:{'sha256':sha(p)} for p in esp.glob('*.npy')}}))
    (esp.parent/'independent_verification.json').write_text(json.dumps(dict(status='COMPLETE_VERIFIED',n_frames=n)))
    save(gas,'frame_index',np.arange(n));save(gas,'charges_pred',np.ones((n,1)))
    (gas/'metadata.json').write_text(json.dumps({'provenance':{'checkpoint':{'sha256':'a'*64}}}))
    bounds=[0,11200,12800,14400,16000]
    for role,a,b in zip(('train','validation','test','guard'),bounds[:-1],bounds[1:]):save(gas,'is_'+role,(np.arange(n)>=a)&(np.arange(n)<b))
    ef={}
    for role,a,b,value in [('train',0,11200,2.),('validation',11200,12800,100.)]:
        p=tmp_path/role;p.mkdir();size=b-a
        arrays=dict(z=np.array([1]),pos=pos[a:b],charge=np.ones(size),mm_pos=mm[a:b],mm_charges=mq[a:b],mm_offsets=np.arange(size+1),source_index=np.arange(a,b),energy=np.ones(size)*value,forces=np.ones((size,1,3))*value)
        for k,v in arrays.items():save(p,k,v)
        ef[role]={k:sha(p/(k+'.npy')) for k in ('energy','forces')}
    contract=dict(gas_charge_checkpoint_sha256='a'*64,gas_prediction_sha256=sha(gas/'charges_pred.npy'),coulomb_constant=1.,charge_projection='as_saved',qm_force_includes_dq_dR=True,mm_self_excluded=True,ef_source_sha256=ef)
    cp=tmp_path/'contract.json';cp.write_text(json.dumps(contract))
    args=argparse.Namespace(source=source,esp=esp,gas=gas,contract=cp,train=tmp_path/'train',validation=tmp_path/'validation',output=tmp_path/'output')
    report=prepare(args)
    assert report['scales']['rms']['energy']==2.
    assert not (args.output/'test').exists() and not (args.output/'guard').exists()
    for role in ('train','validation'):
        for k in ('energy','forces'):assert sha(args.output/role/(k+'.npy'))==ef[role][k]
        np.testing.assert_allclose(np.load(args.output/role/'mm_potential.npy'),2.5)
        np.testing.assert_allclose(np.load(args.output/role/'mm_potential_gradient.npy'),np.tile([.65,.4,.4],(11200 if role=='train' else 1600,1)))
    with pytest.raises(FileExistsError):prepare(args)
    args.output=tmp_path/'bad_output';contract['qm_force_includes_dq_dR']=False;cp.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='baseline contract'):prepare(args)
    assert not args.output.exists()
