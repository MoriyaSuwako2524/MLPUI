
import sys, json, torch
from mlpui.data import NpyDataset
from mlpui.training import Trainer, TrainingConfig
family, ck, D, outp = sys.argv[1:5]
st = torch.load(ck, map_location='cpu', weights_only=True)
W = {'energy': 1., 'forces': 1., 'mm_potential': 1., 'mm_potential_gradient': 1.}
tr = Trainer(family, st['model_config'], TrainingConfig(dtype=torch.float32, device='cuda', loss_weights=W, eval_batch_size=16), checkpoint=ck)
out = {'family': family, 'params': sum(p.numel() for p in tr.model.parameters()), 'epoch': st.get('epoch'), 'early_stopping': st.get('early_stopping'),
       'note': 'RMSE of residual = error of E_gas(Q-Chem) + E_charge(QEq) + E_residual vs QM/MM labels'}
for role in ('validation', 'test', 'guard'):
    m = tr.evaluate(NpyDataset(f'{D}/{role}'))['metrics']
    out[role] = {k: dict(rmse=v['rmse'], mae=v['mae']) for k, v in m.items()}
    print(role, {k: round(v['rmse'], 5) for k, v in m.items()}, flush=True)
json.dump(out, open(outp, 'w'), indent=1)
