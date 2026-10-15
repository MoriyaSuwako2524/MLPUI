"""Synthetic HTTP acceptance: upload -> split -> worker train/evaluate -> reload."""
import copy
from io import BytesIO
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
import pytest
import torch
from mlpui.web.server import make_server
from mlpui.web.config import dataset
from mlpui.external_models import load_external_checkpoint
from test_qmmm import config, offline_config, predict
from test_web import wait_job


def arrays():
    nq, nm = [2, 3, 2, 3, 2, 3], [0, 2, 1, 0, 2, 0]
    qpos = np.concatenate([np.array([[0.,0.,0.],[1.,.1,0.],[0.,1.,.2]])[:n] for n in nq])
    mpos = np.concatenate([np.array([[3.,1.,0.],[-2.,1.,.3]])[:n] for n in nm])
    return dict(z=np.ones(sum(nq), dtype=int), pos=qpos, offsets=np.r_[0,np.cumsum(nq)],
        mm_pos=mpos, mm_charges=np.full(sum(nm),.2), mm_offsets=np.r_[0,np.cumsum(nm)],
        energy=np.arange(6)*.1, forces=np.zeros_like(qpos), mm_forces=np.zeros_like(mpos),
        mm_potential=np.zeros(sum(nm)), frame_index=np.arange(100,106),
        source_index=np.arange(200,206), short_contact=np.array([0,1,0,0,1,0]))


@pytest.mark.parametrize('offline', [False, True])
@pytest.mark.parametrize('gesp', [False, True])
def test_http_qmmm_end_to_end(tmp_path, offline, gesp):
    server=make_server(tmp_path/'runs',0)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}'
    def post(route, value):
        with urlopen(Request(base+route,json.dumps(value).encode(),{'Content-Type':'application/json'})) as response:
            return json.load(response)
    def download(route):
        with urlopen(base+route) as response:
            return response.read()
    try:
        c=offline_config() if offline else config()
        values=arrays()
        props=('energy','forces','mm_forces','mm_potential')
        if gesp:
            props += ('mm_potential_gradient',)
            values['mm_potential_gradient']=np.zeros_like(values['mm_pos'])
        if offline:
            values.update({'baseline_'+p:np.ones_like(values[p])*.02 for p in props})
        entry=post('/api/datasets',{'name':'Synthetic dual region'})
        route='/api/datasets/'+entry['id']
        uploads={}
        for key,value in values.items():
            stream=BytesIO(); np.save(stream,value); uploads[key+'.npy']=stream.getvalue()
        if offline:
            uploads['baseline_metadata.json']=json.dumps(c['baseline']).encode()
        for name,content in uploads.items():
            with urlopen(Request(base+route+'/files/'+name,content,{'Content-Type':'application/octet-stream'})) as response:
                assert response.status==201
        record=post(route+'/finalize',{'files':{k:k+'.npy' for k in values}})
        # Server-directory import is a second supported entry path.
        imported=post('/api/datasets',{'name':'Imported copy','spec':record['spec']})
        assert imported['summary']['samples']==6
        split=post(route+'/split',{'name':'Synthetic split','ratios':[3,1,2],'method':'ordered'})
        specs={d['split_role']:d['spec'] for d in split['datasets']}
        source=dataset(record['spec'])
        start=0
        for role in ('train','validation','test'):
            part=dataset(specs[role])
            assert part.ragged and part.mm_ragged
            for row in range(len(part)):
                for key,value in source[start+row].items():
                    if key=='baseline_metadata': assert part[row][key]==value
                    else: np.testing.assert_array_equal(part[row][key],value)
            start+=len(part)
        payload=dict(name='Synthetic QMMM',family='tensornet_qmmm',model_config=c,
            training=dict(epochs=1,dtype='float64',loss_weights={p:1. for p in props}),**specs)
        assert post('/api/preview',payload)['train']['first_mm_points']==0
        # Missing selected targets must fail before any worker job is created.
        bad=copy.deepcopy(payload); del bad['train']['files']['mm_forces']
        with pytest.raises(HTTPError) as error: post('/api/jobs',bad)
        assert 'missing labels' in error.value.read().decode()
        assert server.manager.list()==[]
        if offline:
            bad=copy.deepcopy(payload); del bad['train']['files']['baseline_forces']
            with pytest.raises(HTTPError) as error: post('/api/jobs',bad)
            assert 'baseline_forces' in error.value.read().decode()
            assert server.manager.list()==[]
        job=post('/api/jobs',payload)
        state=wait_job(server.manager,job['id'])
        assert state['status']=='completed',state
        checkpoint=tmp_path/'downloaded.pt'
        checkpoint.write_bytes(download('/api/jobs/'+job['id']+'/model'))
        loaded=load_external_checkpoint(checkpoint,device='cpu',dtype=torch.float64).model
        evaluation=dict(name='Synthetic evaluation',family='tensornet_qmmm',model_config=c,
            training=payload['training'],task_type='evaluation',evaluation=specs['test'],checkpoint=str(checkpoint))
        job=post('/api/jobs',evaluation); state=wait_job(server.manager,job['id'])
        assert state['status']=='completed',state
        prefix='/api/jobs/'+job['id']+'/artifacts/'
        manifest=json.loads(download(prefix+'metadata.json'))
        assert manifest['properties']['mm_forces']['unit']=='eV/angstrom'
        assert manifest['properties']['mm_potential']['unit']=='eV/e'
        assert manifest['baseline']==(c['baseline'] if offline else None)
        def artifact(name): return np.load(BytesIO(download(prefix+name)),allow_pickle=False)
        np.testing.assert_array_equal(artifact('mm_structure_offsets.npy'),[0,2,2])
        np.testing.assert_array_equal(artifact('qm_counts.npy'),[2,3])
        np.testing.assert_array_equal(artifact('frame_index.npy'),[104,105])
        np.testing.assert_array_equal(artifact('source_index.npy'),[204,205])
        np.testing.assert_array_equal(artifact('short_contact.npy'),[1,0])
        assert state['evaluation']['metrics']['mm_forces']['count']==6
        assert download('/api/jobs/'+job['id']+'/plots/mm_forces.png').startswith(b'\x89PNG')
        predictions=[predict(loaded,dataset(specs['test'])[i],props) for i in range(2)]
        if gesp:
            assert manifest['properties']['mm_potential_gradient']['unit']=='eV/(e*angstrom)'
            assert download('/api/jobs/'+job['id']+'/plots/mm_potential_gradient.png').startswith(b'\x89PNG')
        for p in props:
            expected=np.array([v[p].item() for v in predictions]) if p=='energy' else np.concatenate([v[p].detach().numpy() for v in predictions])
            np.testing.assert_allclose(artifact(p+'_pred.npy').reshape(-1),expected.reshape(-1),atol=1e-10,rtol=1e-10)
        # Validation consists entirely of zero-MM frames: valid empty metrics/artifacts.
        evaluation['evaluation']=specs['validation']
        job=post('/api/jobs',evaluation); state=wait_job(server.manager,job['id'])
        assert state['status']=='completed',state
        assert state['evaluation']['metrics']['mm_forces']==dict(count=0,mae=None,mse=None,rmse=None)
        assert 'mm_forces' not in state['evaluation']['plots']
    finally:
        server.shutdown(); server.manager.close(); server.server_close(); server.lease.close()
