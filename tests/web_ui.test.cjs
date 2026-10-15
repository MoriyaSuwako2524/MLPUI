const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function dashboard() {
  const elements = new Map();
  const document = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      value: id === 'metric' ? 'energy' : '', innerHTML: '', textContent: '',
      panel: {}, closest() { return this.panel; },
      listeners: {}, addEventListener(name, fn) { this.listeners[name] = fn; }, querySelectorAll() { return []; }, removeAttribute(name) { delete this[name]; },
    });
    return elements.get(id);
  }};
  const context = vm.createContext({document, window: {addEventListener() {}}});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../mlpui/web/static/app.js'), 'utf8'), context);
  return {context, element: id => document.getElementById(id), render(job) {
    context.fixture = job;
    vm.runInContext('jobs=[fixture]; renderJobs(); renderDetail(fixture);', context);
  }};
}

function job(extra = {}) {
  return {id: 'a'.repeat(32), name: 'Saved run', family: 'newtonnet',
    status: 'completed', created: 1, epochs: 10, completed: 2,
    history: [{epoch: 1, train: {energy: 0.2}}],
    summary: {train: {samples: 8}}, ...extra};
}

test('legacy training renders its epochs, loss chart and checkpoints', () => {
  const ui = dashboard();
  ui.render(job({checkpoints: ['epoch_000001.pt']}));
  assert.match(ui.element('job-list').innerHTML, /Training/);
  assert.match(ui.element('job-list').innerHTML, /1 \/ 10 epochs/);
  assert.match(ui.element('detail-meta').textContent, /8  training structures/);
  assert.match(ui.element('chart').innerHTML, /<svg/);
  assert.equal(ui.element('chart').panel.hidden, false);
  assert.equal(ui.element('checkpoint-list').panel.hidden, false);
  assert.match(ui.element('checkpoint-list').innerHTML, /epoch_000001.pt/);
  assert.equal(ui.element('evaluation-panel').hidden, true);
  assert.equal(ui.element('prediction-panel').hidden, true);
});

for (const type of ['evaluation', 'prediction']) {
  for (const explicit of [true, false]) {
    test(`${type} renders with ${explicit ? 'explicit' : 'inferred'} task type`, () => {
      const ui = dashboard();
      ui.render(job({task_type: explicit ? type : undefined, status: 'running',
        summary: {[type]: {samples: 8}}}));
      assert.match(ui.element('job-list').innerHTML, /2 \/ 8 structures/);
      assert.equal(ui.element('progress').value, 25);
      assert.equal(ui.element('detail-status').textContent, type === 'prediction' ? 'Predicting' : 'Evaluating');
      assert.equal(ui.element('chart').panel.hidden, true);
      assert.equal(ui.element('evaluation-panel').hidden, type !== 'evaluation');
      assert.equal(ui.element('prediction-panel').hidden, type !== 'prediction');
    });
  }
}

for (const type of ['training', 'evaluation', 'prediction']) {
  test(`${type} tolerates missing sample metadata`, () => {
    const ui = dashboard();
    for (const summary of [undefined, {}, {[type === 'training' ? 'train' : type]: {}}]) {
      ui.render(job({task_type: type, summary, history: undefined}));
      assert.match(ui.element('detail-meta').textContent, /—  .* structures/);
      if (type !== 'training') {
        assert.match(ui.element('job-list').innerHTML, /2 \/ — structures/);
        assert.equal(ui.element('progress').value, undefined);
      }
    }
  });
}

test('QMMM form preserves target selection, mapping and offline identity', () => {
  const ui=dashboard();
  ui.context.structuredClone=structuredClone;
  ui.context.document.querySelectorAll=()=>[];
  vm.runInContext("let datasetRecords=[]; backends={tensornet_qmmm:{training_targets:['energy','forces','mm_forces','mm_potential']}};",ui.context);
  const values={'family':'tensornet_qmmm','task-type':'evaluation','model-config':'{}',
    'qmmm-baseline-mode':'offline','qmmm-baseline-metadata':JSON.stringify({mode:'offline',source_manifest_sha256:'c'.repeat(64)}),
    'checkpoint':'downloaded.pt','energy-scale':'1','length-scale':'1','dtype':'float64','device':'cpu'};
  for(const [key,value] of Object.entries(values))ui.element(key).value=value;
  for(const key of ['z','pos','mm_pos','mm_charges','mm_offsets','mm_forces','baseline_energy','baseline_mm_forces'])ui.element('file-'+key).value=key+'.npy';
  ui.element('eval-mm_forces').checked=true;
  vm.runInContext('updateBackendTargets(); result=payload();',ui.context);
  assert.equal(ui.element('qmmm-settings').hidden,false);
  assert.equal(ui.element('qmmm-baseline-label').hidden,false);
  assert.equal(ui.element('weight-mm_forces').disabled,true);
  assert.equal(ui.element('eval-charges').disabled,true);
  const result=JSON.parse(JSON.stringify(ui.context.result));
  assert.deepEqual(result.training.loss_weights,{mm_forces:1});
  assert.equal(result.evaluation.files.mm_offsets,'mm_offsets.npy');
  assert.equal(result.model_config.baseline.mode,'offline');
  ui.element('file-mm_forces').value='';
  assert.throws(()=>vm.runInContext('payload()',ui.context),/missing mm_forces labels/);
});

test('zero-MM evaluation displays empty metrics, units and downloads',()=>{
  const ui=dashboard();
  ui.render(job({family:'tensornet_qmmm',task_type:'evaluation',evaluation:{metrics:{mm_forces:{count:0,mae:null,mse:null,rmse:null}},plots:{},artifacts:{manifest:'metadata.json',files:{'mm_counts.npy':{}},properties:{mm_forces:{unit:'eV/angstrom'}}}}}));
  assert.match(ui.element('evaluation-metrics').innerHTML,/N\/A \(no components\)/);
  assert.match(ui.element('evaluation-metrics').innerHTML,/eV\/angstrom/);
  assert.match(ui.element('evaluation-metrics').innerHTML,/<td>0<\/td>/);
  assert.match(ui.element('evaluation-artifacts').innerHTML,/mm_counts.npy/);
});

test('GESP selection survives UI payload and requires its own labels',()=>{
  const ui=dashboard();
  ui.context.structuredClone=structuredClone;
  ui.context.document.querySelectorAll=()=>[];
  vm.runInContext("let datasetRecords=[]; backends={tensornet_qmmm:{training_targets:['energy','forces','mm_potential','mm_potential_gradient']}};",ui.context);
  for(const [key,value] of Object.entries({'family':'tensornet_qmmm','task-type':'evaluation','model-config':'{}','qmmm-baseline-mode':'residual_only','checkpoint':'downloaded.pt','energy-scale':'1','length-scale':'1','dtype':'float64','device':'cpu'}))ui.element(key).value=value;
  for(const key of ['z','pos','mm_pos','mm_charges','mm_offsets','mm_potential_gradient'])ui.element('file-'+key).value=key+'.npy';
  ui.element('eval-mm_potential_gradient').checked=true;
  vm.runInContext('updateBackendTargets(); result=payload();',ui.context);
  assert.deepEqual(JSON.parse(JSON.stringify(ui.context.result.training.loss_weights)),{mm_potential_gradient:1});
  assert.equal(ui.context.result.evaluation.files.mm_potential_gradient,'mm_potential_gradient.npy');
  ui.element('file-mm_potential_gradient').value='';
  assert.throws(()=>vm.runInContext('payload()',ui.context),/missing mm_potential_gradient labels/);
});


test('switching to QMMM after selecting a managed dataset restores offline identity',()=>{
  const ui=dashboard();
  ui.context.document.querySelectorAll=()=>[];
  vm.runInContext('init=()=>{};',ui.context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../mlpui/web/static/datasets.js'),'utf8'),ui.context);
  vm.runInContext("models={tensornet_qmmm:{baseline:{mode:'residual_only'}}}; backends={tensornet_qmmm:{training_targets:['energy','forces','mm_forces','mm_potential']}}; datasetRecords=[{id:'selected',spec:{directory:'data'},summary:{fields:['z','pos','mm_pos','mm_charges'],baseline_metadata:{mode:'offline',source_manifest_sha256:'identity'}}}];",ui.context);
  ui.element('family').value='tensornet_qmmm';
  ui.element('managed-train').value='selected';
  ui.element('task-type').value='training';
  ui.element('family').listeners.change();
  assert.equal(ui.element('qmmm-baseline-mode').value,'offline');
  assert.equal(JSON.parse(ui.element('qmmm-baseline-metadata').value).source_manifest_sha256,'identity');
  assert.equal(ui.element('directory').value,'data');
  assert.equal(ui.element('qmmm-baseline-label').hidden,false);
});
