"""Fixed-response GESP is not a total mixed charge/position derivative."""
import copy
import numpy as np
import pytest
import torch
from mlpui.backends import get_backend
from mlpui.data import NpyDataset
from mlpui.training import Trainer, TrainingConfig

TARGETS=('energy','forces','mm_potential','mm_potential_gradient')

def config():
    c=get_backend('tensornet_qmmm').default_config()
    c.update(embedding_dimension=8,num_layers=1,num_rbf=4,precision=64)
    return c

def sample(charges=(0.,1.e-12,-.3)):
    return dict(z=np.array([8,1,1]),pos=np.array([[0.,0.,0.],[1.,.2,.1],[-.2,.9,.3]]),
                mm_pos=np.array([[2.,-.7,.5],[-1.2,1.8,-.6],[2.,2.,1.]])[:len(charges)],
                mm_charges=np.array(charges))

def predict(model,s,props=TARGETS):
    b=get_backend('tensornet_qmmm')
    return b.forward(model,b.prepare_sample(s,'cpu',torch.float64,model=model),props)

def test_fixed_response_gradient_finite_difference_and_not_mixed():
    from mlpui.models.tensornet_qmmm.model import environment, fixed_response_mm_properties
    p=torch.tensor([[0.,0.,0.],[.2,1.,.1]],dtype=torch.float64)
    m=torch.tensor([[2.,.3,-.7]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([.3],dtype=torch.float64,requires_grad=True)
    phi,field=environment(p,m,q)
    energy=.5*phi.square().sum()+.2*field.square().sum()
    a,b=torch.autograd.grad(energy,(phi,field),create_graph=True,retain_graph=True)
    potential,gradient=fixed_response_mm_properties(p,m,a,b)
    derivative=torch.autograd.grad(energy,q,create_graph=True,retain_graph=True)[0]
    torch.testing.assert_close(potential,derivative)
    force=-torch.autograd.grad(energy,m,retain_graph=True)[0]
    torch.testing.assert_close(force,-q[:,None]*gradient)
    mixed=torch.autograd.grad(derivative.sum(),m)[0]
    assert not torch.allclose(mixed,gradient,rtol=1e-3,atol=1e-3)
    step=1e-5
    for axis in range(3):
        plus=m.detach().clone();minus=plus.clone();plus[0,axis]+=step;minus[0,axis]-=step
        vp,_=fixed_response_mm_properties(p,plus,a.detach(),b.detach())
        vm,_=fixed_response_mm_properties(p,minus,a.detach(),b.detach())
        torch.testing.assert_close((vp-vm)/(2*step),gradient[:,axis],rtol=1e-8,atol=1e-8)

def test_real_model_zero_small_charge_identity_and_each_target_backward():
    torch.manual_seed(11)
    model=get_backend('tensornet_qmmm').build_model(config()).train()
    out=predict(model,sample(),TARGETS+('mm_forces',))
    torch.testing.assert_close(out['mm_forces'],-torch.tensor(sample()['mm_charges'])[:,None]*out['mm_potential_gradient'],rtol=1e-9,atol=1e-9)
    assert out['mm_potential_gradient'][0].abs().max()>0
    for key in TARGETS:
        model.zero_grad(set_to_none=True)
        value=predict(model,sample())[key]
        (value.square().mean()+value.mean()).backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()),key
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_gesp_does_not_change_energy_derivative_contract():
    model=get_backend('tensornet_qmmm').build_model(config()).eval()
    s=sample();out=predict(model,s,TARGETS+('mm_forces',))
    legacy=predict(model,s,('energy','forces','mm_forces','mm_potential'))
    for key in legacy:
        if key!='components':torch.testing.assert_close(out[key],legacy[key],rtol=0,atol=0)
    # Actual energy finite differences include all response/geometry dependence.
    for field,index,prop,sign in [('pos',(1,2),'forces',-1),
                                  ('mm_pos',(2,0),'mm_forces',-1),
                                  ('mm_charges',(0,),'mm_potential',1)]:
        plus=copy.deepcopy(s);minus=copy.deepcopy(s)
        plus[field][index]+=1e-5;minus[field][index]-=1e-5
        fd=(predict(model,plus,('energy',))['energy']-predict(model,minus,('energy',))['energy'])/2e-5
        torch.testing.assert_close(fd,sign*out[prop][index],rtol=1e-6,atol=1e-7)


def test_gesp_rotates_as_spatial_gradient():
    model=get_backend('tensornet_qmmm').build_model(config()).eval()
    s=sample();out=predict(model,s)
    rotation=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])  # includes reflection
    transformed=copy.deepcopy(s)
    for key in ('pos','mm_pos'):transformed[key]=s[key]@rotation.T+np.array([3.,-1.,2.])
    actual=predict(model,transformed)
    torch.testing.assert_close(actual['mm_potential'],out['mm_potential'],rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(actual['mm_potential_gradient'],out['mm_potential_gradient']@torch.tensor(rotation.T),rtol=1e-9,atol=1e-9)

def test_dataset_gesp_ragged_unit_no_force_sign(tmp_path):
    arrays=dict(z=[1],pos=np.zeros((2,1,3)),mm_pos=np.array([[2.,0.,0.],[3.,0.,0.]]),
                mm_charges=[0.,1e-12],mm_offsets=[0,0,2],mm_potential_gradient=np.ones((2,3)))
    for k,v in arrays.items():np.save(tmp_path/(k+'.npy'),v)
    d=NpyDataset(tmp_path,gradients=True,energy_scale=3.,length_scale=2.)
    assert d[0]['mm_potential_gradient'].shape==(0,3)
    np.testing.assert_allclose(d[1]['mm_potential_gradient'],1.5)

def write_data(path):
    path.mkdir();s=sample()
    arrays=dict(z=s['z'],pos=np.stack([s['pos'],s['pos']+.1]),mm_pos=s['mm_pos'],
                mm_charges=s['mm_charges'],mm_offsets=[0,0,3],energy=[.1,.2],
                forces=np.ones((2,3,3))*.03,mm_potential=np.array([.1,.2,.3]),
                mm_potential_gradient=np.ones((3,3))*.02)
    for k,v in arrays.items():np.save(path/(k+'.npy'),v)
    return NpyDataset(path)

def test_four_target_frame_normalization_checkpoint_and_evaluation(tmp_path):
    data=write_data(tmp_path/'data')
    tc=TrainingConfig(epochs=1,batch_size=2,dtype=torch.float64,
                      loss_weights=dict(zip(TARGETS,[1.,2.,3.,4.])),early_stopping=True)
    t=Trainer('tensornet_qmmm',config(),tc)
    # Duplicating MM labels must not increase the per-frame MM loss just by count.
    t.model.eval();s=data[1]
    _,parts=t._loss(s)
    pred=predict(t.model,s)
    for k in ('mm_potential','mm_potential_gradient'):
        torch.testing.assert_close(parts[k],(pred[k]-torch.tensor(s[k])).square().mean())
    duplicated=copy.deepcopy(s)
    for k in ('mm_pos','mm_potential','mm_potential_gradient'):
        duplicated[k]=np.concatenate([s[k],s[k]])
    duplicated['mm_charges']=np.tile(s['mm_charges']/2,2)
    _,duplicate_parts=t._loss(duplicated)
    for k in TARGETS:
        torch.testing.assert_close(parts[k],duplicate_parts[k])
    h=t.fit(data,data,output_dir=tmp_path/'run')
    assert set(h[0]['validation'])==set(TARGETS)
    # Fit/early stopping gives each structure equal weight, including empty MM.
    t.model.eval()
    for k in TARGETS:
        expected=sum(t._loss(data[i])[1][k].item() for i in range(2))/2
        assert h[0]['validation'][k]==pytest.approx(expected)
    restored=Trainer('tensornet_qmmm',config(),tc,checkpoint=tmp_path/'run/best.pt')
    t.model.eval();restored.model.eval()
    for k,v in predict(t.model,s).items():
        if k in TARGETS:torch.testing.assert_close(v,predict(restored.model,s)[k])
    result=restored.evaluate(data,artifact_dir=tmp_path/'artifacts',plot_dir=tmp_path/'plots',artifact_metadata={'model_family':'tensornet_qmmm'})
    assert result['metrics']['mm_potential_gradient']['count']==9
    assert result['artifacts']['properties']['mm_potential_gradient']['unit']=='eV/(e*angstrom)'
    assert np.array_equal(np.load(tmp_path/'artifacts/mm_structure_offsets.npy'),[0,0,3])

def test_web_targets_and_positive_weights():
    assert 'mm_potential_gradient' in get_backend('tensornet_qmmm').training_targets
    with pytest.raises(ValueError,match='positive'):
        TrainingConfig(loss_weights={'mm_potential_gradient':0.})
