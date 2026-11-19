
import sys, json, torch
from mlpui.data import NpyDataset
from mlpui.training import Trainer, TrainingConfig
ck, D = sys.argv[1], sys.argv[2]
st = torch.load(ck, map_location='cpu', weights_only=True)
W = {'energy': 1., 'forces': 1., 'mm_potential': 1., 'mm_potential_gradient': 1., 'gas_mm_potential': 1., 'gas_mm_potential_gradient': 1.}
tr = Trainer('qeq_charge', st['model_config'], TrainingConfig(dtype=torch.float64, device='cuda', loss_weights=W, eval_batch_size=64), checkpoint=ck)
out = {'epoch': st.get('epoch'), 'early_stopping': st.get('early_stopping')}
for role in ('validation', 'test', 'guard'):
    out[role] = {k: v['rmse'] for k, v in tr.evaluate(NpyDataset(f'{D}/{role}'))['metrics'].items()}
    print(role, out[role], flush=True)
json.dump(out, open(sys.argv[3], 'w'), indent=1)
