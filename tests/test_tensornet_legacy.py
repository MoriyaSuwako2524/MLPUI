"""Regression against the exact pre-replacement implementation (f753667)."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from mlpui.backends import get_backend, find_backend
from mlpui.external_models import load_external_checkpoint
from test_qmmm import sample, offline_config

PROPS = ('energy', 'forces', 'mm_potential', 'mm_potential_gradient')


def config():
    return {**get_backend('tensornet-ext').default_config(), 'precision': 64,
            'embedding_dimension': 8, 'num_layers': 1, 'num_rbf': 4}


def recovered():
    spec = importlib.util.spec_from_file_location('schema1_reference', Path(__file__).parent/'fixtures/qmmm_schema1_f753667.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TensorNetQMMM


def inputs(model, value=None):
    return get_backend('tensornet-ext').prepare_sample(value or sample(), 'cpu', torch.float64, model=model)


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('mm_count', [0, 2])
def test_exact_historical_numerics_and_four_target_gradients(training, mm_count):
    torch.manual_seed(17)
    old = recovered()(config()).train(training)
    model = get_backend('tensornet-ext').build_model(config()).train(training)
    model.load_state_dict(old.state_dict())
    s = sample(); s['mm_pos'] = s['mm_pos'][:mm_count]; s['mm_charges'] = s['mm_charges'][:mm_count]
    if mm_count: s['mm_charges'][0] = 0.  # zero-charge fixed-response limit
    actual = model(**inputs(model,s), properties=PROPS)
    expected = old(**inputs(model,s), properties=PROPS)
    for p in PROPS:
        torch.testing.assert_close(actual[p], expected[p], rtol=0, atol=0)
    if training:
        for p in PROPS:
            model.zero_grad(); old.zero_grad()
            actual[p].square().sum().backward(retain_graph=True)
            expected[p].square().sum().backward(retain_graph=True)
            for a,b in zip(model.parameters(),old.parameters()):
                if a.grad is None: assert b.grad is None
                else:
                    assert torch.isfinite(a.grad).all()
                    torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


@pytest.mark.parametrize('family', ['tensornet-ext', 'tensornet_qmmm'])
def test_checkpoint_roundtrip_and_cross_architecture_rejection(tmp_path, family):
    from test_qmmm import config as current_config
    backend = get_backend(family)
    c = config() if family == 'tensornet-ext' else current_config()
    model = backend.build_model(c).eval()
    checkpoint = tmp_path/'model.pt'
    torch.save(dict(format_version=1, backend=family, model_config_version=1,
                    model_config=c,state_dict=model.state_dict()), checkpoint)
    restored = load_external_checkpoint(checkpoint, device='cpu').model
    assert restored.mlpui_family == family
    assert find_backend(restored.state_dict()).name == family
    for prop in PROPS:
        torch.testing.assert_close(model(**inputs(model),properties=PROPS)[prop],
                                   restored(**inputs(restored),properties=PROPS)[prop],rtol=0,atol=0)
    other = 'tensornet_qmmm' if family == 'tensornet-ext' else 'tensornet-ext'
    with pytest.raises(ValueError): load_external_checkpoint(checkpoint,family=other)
    with pytest.raises(ValueError): get_backend(other).build_model(c)
    target = get_backend(other).build_model(current_config() if other=='tensornet_qmmm' else config())
    with pytest.raises(ValueError): target.load_state_dict(model.state_dict(),strict=False)
    torch.save(model,checkpoint)
    assert load_external_checkpoint(checkpoint,trusted_checkpoint=True,device='cpu').model.mlpui_family == family


def test_explicit_schema1_checkpoint_migration(tmp_path):
    c = config(); c.pop('architecture')
    old = recovered()(c)
    checkpoint = tmp_path/'historical.pt'
    torch.save(dict(format_version=1,backend='tensornet_qmmm',model_config=c,state_dict=old.state_dict()),checkpoint)
    with pytest.raises(ValueError): load_external_checkpoint(checkpoint)
    restored = load_external_checkpoint(checkpoint,family='tensornet-ext',device='cpu').model
    assert restored.config['architecture']=='tensornet_qm_h3_v1'
    old.eval()
    for p in PROPS:
        torch.testing.assert_close(restored(**inputs(restored),properties=PROPS)[p],
                                   old(**inputs(old),properties=PROPS)[p],rtol=0,atol=0)
    # Forging metadata cannot make incompatible weights pass strict loading.
    bad=old.state_dict(); bad['qmmm_schema']=torch.tensor(2)
    torch.save(dict(backend='tensornet_qmmm',model_config=c,state_dict=bad),checkpoint)
    with pytest.raises(ValueError): load_external_checkpoint(checkpoint,family='tensornet-ext')


@pytest.mark.parametrize('offline', [False, True])
def test_ragged_batch_and_baseline_reconstruction(offline):
    c=config()
    if offline: c['baseline']=offline_config()['baseline']
    model=get_backend('tensornet-ext').build_model(c).train()
    samples=[]
    for n in (0,2):
        s=sample(); s['mm_pos']=s['mm_pos'][:n]; s['mm_charges']=s['mm_charges'][:n]
        if offline:
            s['baseline_metadata']=c['baseline']
            for p in PROPS:
                shape=() if p=='energy' else (3,3) if p=='forces' else (n,) if p=='mm_potential' else (n,3)
                s['baseline_'+p]=np.full(shape,.125)
        samples.append(s)
    rows=[model(**inputs(model,s),properties=PROPS) for s in samples]
    joined={k:np.concatenate([s[k] for s in samples]) for k in ('z','pos','mm_pos','mm_charges')}
    joined.update(batch=np.repeat([0,1],3),mm_offsets=np.array([0,0,2]),charge=np.zeros(2))
    if offline:
        joined['baseline_metadata']=c['baseline']
        for p in PROPS:
            joined['baseline_'+p]=(np.array if p=='energy' else np.concatenate)([s['baseline_'+p] for s in samples])
    result=model(**inputs(model,joined),properties=PROPS)
    for p in PROPS:
        expected=(torch.stack if p=='energy' else torch.cat)([r[p] for r in rows])
        torch.testing.assert_close(result[p],expected,rtol=0,atol=0)
    sum(result[p].square().sum() for p in PROPS).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if offline:
        residual=get_backend('tensornet-ext').build_model(config()).train()
        residual.load_state_dict(model.state_dict())
        for s,row in zip(samples,rows):
            bare={k:v for k,v in s.items() if not k.startswith('baseline_')}
            prediction=residual(**inputs(residual,bare),properties=PROPS)
            for p in PROPS: torch.testing.assert_close(row[p],prediction[p]+.125)


def test_web_presets_preserve_comparison_identity():
    from mlpui.web.config import presets, backend_metadata, normalize
    metadata=backend_metadata()
    weights=dict(zip(PROPS,(1.,2.,3.,4.)))
    for family in ('tensornet-ext','tensornet_qmmm'):
        c=presets()[family]
        result=normalize(dict(family=family,model_config=c,training=dict(loss_weights=weights,seed=37),
                              train={'directory':'same-training-records','energy_scale':1.,'length_scale':1.}))
        assert result['training']==dict(loss_weights=weights,seed=37)
        assert result['model_config']['architecture']==metadata[family]['architecture']
        assert 'H+3' in metadata[family]['config_hint']
        assert set(PROPS)<=set(metadata[family]['training_targets'])


def test_interleaved_ragged_regions_preserve_atom_order():
    model=get_backend('tensornet-ext').build_model(config()).eval()
    s=sample()
    joined={k:np.concatenate([s[k],s[k]]) for k in ('z','pos','mm_pos','mm_charges')}
    joined.update(batch=np.repeat([0,1],3),mm_batch=np.repeat([0,1],2),charge=np.zeros(2))
    grouped=model(**inputs(model,joined),properties=PROPS)
    qp=np.array([3,0,4,1,5,2]); mp=np.array([2,0,3,1])
    for k in ('z','pos','batch'): joined[k]=joined[k][qp]
    for k in ('mm_pos','mm_charges','mm_batch'): joined[k]=joined[k][mp]
    mixed=model(**inputs(model,joined),properties=PROPS)
    for p in PROPS:
        expected=grouped[p] if p=='energy' else grouped[p][mp if p.startswith('mm_') else qp]
        torch.testing.assert_close(mixed[p],expected,rtol=0,atol=0)


def test_reference_source_provenance():
    import hashlib
    source=(Path(__file__).parent/'fixtures/qmmm_schema1_f753667.py').read_text(encoding='utf-8')
    assert hashlib.sha256(source.encode()).hexdigest() == '55d111d9e0b878ec46860a82bfb54f1534dbae481782e58db3774ea92a2f0931'
