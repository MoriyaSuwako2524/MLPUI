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
      addEventListener() {}, removeAttribute(name) { delete this[name]; },
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
