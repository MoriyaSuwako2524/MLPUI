const $ = id => document.getElementById(id);
const active = new Set(['queued','starting','running','stopping']);
const statuses = {queued:'Queued',cancelled:'Cancelled',starting:'Starting',running:'Training',completed:'Completed',stopped:'Stopped',failed:'Failed',interrupted:'Interrupted'};
let jobs = [], models = {}, backends = {}, current = null, pollBusy = false;
const backendLabel = family => backends[family]?.label || family;
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// Training records created before task types were introduced omit task_type.
const jobType = job => job.task_type || (job.summary?.prediction ? 'prediction' : job.summary?.evaluation ? 'evaluation' : 'training');
function jobSamples(job) {
  const type=jobType(job), samples=job.summary?.[type==='training'?'train':type]?.samples;
  return Number.isFinite(samples)&&samples>=0 ? samples : null;
}
function notify(message='') { $('notice').textContent=message; $('notice').hidden=!message; }
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {cache:'no-store'} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const result = await response.json();
  if(!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
function text(id) { return $(id).value.trim(); }
function number(id) { const v=Number(text(id)); if(!text(id)||!Number.isFinite(v)) throw new Error('Enter a valid number'); return v; }
function groups(id) { return text(id).split(',').map(v=>v.trim()).filter(Boolean); }
function predictionPayload(){
  if(!text('checkpoint')) throw new Error('UMA prediction requires a checkpoint path');
  const selected=datasetRecords.find(d=>d.id===text('managed-train'));
  let model; try {model=JSON.parse(text('model-config'));} catch {throw new Error('Invalid UMA configuration JSON');}
  const files={};
  for(const key of ['z','pos','cell','pbc','offsets','charge','spin']) if(text('file-'+key)) files[key]=text('file-'+key);
  const spec=selected?structuredClone(selected.spec):{directory:text('directory'),files,length_scale:number('length-scale')};
  if(!selected&&groups('shards').length) spec.shards=groups('shards');
  return {task_type:'prediction',family:'uma',name:text('name'),checkpoint:text('checkpoint'),model_config:model,
    prediction:spec,training:{device:text('device'),dtype:text('dtype')}};
}
function payload() {
  if(text('task-type')==='prediction') return predictionPayload();
  const evaluating=text('task-type')!=='training';
  const selected=datasetRecords.find(d=>d.id===text('managed-train'));
  const weights={};
  for(const key of ['energy','forces','charges']) {
    if(!backends[text('family')]?.training_targets.includes(key)) continue;
    if(evaluating) {if($('eval-'+key).checked) weights[key]=1;}
    else if((key!=='charges'||$('train-charges').checked)&&number('weight-'+key)>0) weights[key]=number('weight-'+key);
  }
  if(!Object.keys(weights).length) throw new Error('Select at least one target: energy, forces or charges');
  const files={};
  for(const key of ['z','pos','energy','forces','charges','cell','pbc','offsets']) {
    if(['energy','forces','charges'].includes(key) && !(key in weights)) continue;
    if(text('file-'+key)) files[key]=text('file-'+key);
  }
  for(const key of Object.keys(weights)) if(selected?!selected.summary.fields.includes(key):!files[key]) throw new Error('Dataset is missing '+key+' labels');
  const train=selected?{...structuredClone(selected.spec),dataset_id:selected.id,dataset_name:selected.name}:{directory:text('directory'),files,gradients:$('gradients').checked,energy_scale:number('energy-scale'),length_scale:number('length-scale')};
  if(!selected&&groups('shards').length) train.shards=groups('shards');
  let model;
  try { model=JSON.parse($('model-config').value); } catch { throw new Error('Model configuration is not valid JSON'); }
  const container=backends[text('family')]?.config_container;
  const modelSettings=container&&model[container]&&typeof model[container]==='object'?model[container]:model;
  if($('charge-constraint').checked) {
    if(!('charges' in weights)) throw new Error('Also enable atomic-charge training or evaluation');
    modelSettings.charge_constraint=true;
  }
  if(modelSettings.charge_constraint) {
    if(selected?!selected.summary.fields.includes('charge'):!text('file-charge')) throw new Error('The hard constraint requires total charge Q for each structure');
    if(!selected) files.charge=text('file-charge');
  }
  if(evaluating) {
    if(!text('checkpoint')) throw new Error('Evaluation requires an existing model path');
    return {task_type:'evaluation',name:text('name'),family:text('family'),model_config:model,
      checkpoint:text('checkpoint'),evaluation:train,
      training:{dtype:text('dtype'),device:text('device'),loss_weights:weights}};
  }
  const result={name:text('name'),family:text('family'),model_config:model,train,
    training:{epochs:number('epochs'),batch_size:number('batch-size'),learning_rate:number('learning-rate'),dtype:text('dtype'),device:text('device'),seed:number('seed'),loss_weights:weights,save_interval:number('save-interval'),max_checkpoints:number('max-checkpoints'),test_interval:number('test-interval')}};
  if(text('training-mode')==='continue') {
    if(!text('checkpoint')) throw new Error('Continuation requires an existing .pt model path');
    result.checkpoint=text('checkpoint');
  }
  if($('early-stopping').checked) Object.assign(result.training,{early_stopping:true,
    early_stopping_monitor:text('early-monitor'),early_stopping_patience:number('early-patience'),
    early_stopping_min_delta:number('early-min-delta')});
  if(text('validation-directory') || groups('validation-shards').length) {
    result.validation={...train,directory:text('validation-directory')||train.directory};
    delete result.validation.shards;
    if(groups('validation-shards').length) result.validation.shards=groups('validation-shards');
  }
  if(text('test-directory') || groups('test-shards').length) {
    result.test={...train,directory:text('test-directory')||train.directory};
    delete result.test.shards;
    if(groups('test-shards').length) result.test.shards=groups('test-shards');
  }
  for(const split of ['validation','test']) {
    const record=datasetRecords.find(d=>d.id===text('managed-'+split));
    if(record) result[split]=structuredClone(record.spec);
  }
  return result;
}
function route() {
  const hash=location.hash || '#jobs';
  const view=hash==='#datasets'?'datasets':hash==='#new'?'new':hash.startsWith('#job/')?'detail':'jobs';
  for(const name of ['jobs','new','detail','datasets']) $(name+'-view').hidden=name!==view;
  $('nav-jobs').classList.toggle('active',view==='jobs'||view==='detail'); $('nav-new').classList.toggle('active',view==='new');
  $('nav-datasets').classList.toggle('active',view==='datasets');
  $('breadcrumb').textContent=view==='datasets'?'Dataset':view==='new'?'New job':view==='detail'?'Job details':'Jobs';
  if(view==='datasets'||view==='new') loadDatasets().catch(error=>notify(error.message));
  notify(); current=null;
  if(view==='detail') {
    $('detail-name').textContent='Loading...'; $('detail-meta').textContent=''; $('log').textContent='Loading...';
    $('stop').hidden=true; $('download-model').hidden=true;
  }
  refresh();
}
function renderJobs() {
  $('count-all').textContent=jobs.length;
  $('count-active').textContent=jobs.filter(j=>active.has(j.status)&&j.status!=='queued').length;
  $('count-queued').textContent=jobs.filter(j=>j.status==='queued').length;
  $('count-done').textContent=jobs.filter(j=>j.status==='completed').length;
  if(!jobs.length) {
    $('job-list').innerHTML='<div class="empty"><div class="empty-icon">▦</div><h2>Start your first job</h2><p>Connect NPY data and configure a model. Each experiment keeps its own records.</p><a class="button primary" href="#new">＋ Create job</a></div>';
    return;
  }
  $('job-list').innerHTML='<div class="table-wrap"><table><thead><tr><th>Job</th><th>Model</th><th>Status</th><th>Progress</th><th>Created</th></tr></thead><tbody>'+jobs.map(j=>`<tr><td><a href="#job/${j.id}">${escapeHTML(j.name)}</a><small>${jobType(j)!=='training'?(jobType(j)==='prediction'?'Prediction':'Evaluation'):'Training'} · ${j.id.slice(0,8)} · ${escapeHTML(j.assigned_device||j.requested_device||'cpu')}${j.status==='queued'?' · Waiting':''}</small></td><td>${escapeHTML(backendLabel(j.family))}</td><td><span class="badge ${escapeHTML(j.status)}">${j.stop_requested&&active.has(j.status)?'Stopping':(jobType(j)!=='training'&&j.status==='running'?(jobType(j)==='prediction'?'Predicting':'Evaluating'):statuses[j.status])||escapeHTML(j.status)}</span></td><td>${jobType(j)!=='training'?`${j.completed||0} / ${jobSamples(j)??'—'} structures`:`${(j.history||[]).length} / ${j.epochs} epochs`}</td><td>${escapeHTML(new Date(j.created*1000).toLocaleString())}</td></tr>`).join('')+'</tbody></table></div>';
}
function renderDetail(job) {
  current=job;
  const predicting=jobType(job)==='prediction';
  const evaluating=jobType(job)!=='training';
  const samples=jobSamples(job);
  $('detail-name').textContent=job.name;
  $('detail-meta').textContent=`${backendLabel(job.family)} · ${samples??'—'}  ${predicting?'prediction':evaluating?'evaluation':'training'} structures · ${job.id.slice(0,8)}`;
  $('detail-status').className='badge '+job.status;
  $('detail-status').textContent=job.stop_requested&&active.has(job.status)?'Stopping':statuses[job.status];
  const partial=job.phase==='training'?(job.completed||0)/(job.total||1):['validation','test'].includes(job.phase)?1:0;
  const epoch=(job.history||[]).length;
  $('progress').value=Math.min(100,100*(epoch+partial)/job.epochs);
  $('progress-label').textContent=`${epoch} / ${job.epochs} epochs`+(job.phase==='training'?` · ${job.completed} / ${job.total} structures`:job.phase==='loading'?' · Loading model and data':job.phase==='validation'?' · Validating':job.phase==='test'?' · Evaluating test data':'');
  if(evaluating) {
    if(samples===null) $('progress').removeAttribute('value');
    else $('progress').value=Math.min(100,100*(job.completed||0)/(samples||1));
    $('progress-label').textContent=`${job.completed||0} / ${samples??'—'} structures`+(job.phase==='loading'?' · Loading model and data':'');
    if(job.status==='running') $('detail-status').textContent=job.phase==='plotting'?'Plotting':'Evaluating';
  }
  $('detail-error').hidden=!job.error; $('detail-error').textContent=job.error||'';
  $('stop').hidden=!active.has(job.status); $('stop').disabled=job.stop_requested;
  $('stop').textContent=job.status==='queued'?'Cancel queued job':job.stop_requested?'Stopping…':'Stop job';
  if(job.status==='queued') $('progress-label').textContent=`Queue position ${job.queue_position||'—'} · ${job.queue_reason||'Waiting for a resource'}`;
  $('detail-meta').textContent+=` · ${job.assigned_device||job.requested_device||'cpu'}`;
  const early=job.early_stopping;
  $('early-stopping-result').hidden=!early;
  $('early-stopping-result').textContent=early?`${early.stopped?'Early stopping triggered':'Early stopping'} · Validation set ${early.monitor} · Best epoch ${early.best_epoch??'—'} · Best value ${early.best_value==null?'—':early.best_value.toExponential(5)} · No sufficient improvement ${early.bad_epochs}/${early.patience} epochs`:'';
  if(job.stop_reason==='early_stopping'){$('detail-status').textContent='Early stopped';$('progress').value=100;$('progress-label').textContent=`${epoch} / ${job.epochs} epochs · Validation did not improve`;}
  $('download-best').hidden=!job.best_checkpoint;
  $('download-best').href=`/api/jobs/${job.id}/best`;
  $('output-path').textContent=job.directory;
  $('download-model').hidden=!job.checkpoint || !['completed','stopped'].includes(job.status);
  $('download-model').href=`/api/jobs/${job.id}/model`;
  $('download-config').href=`/api/jobs/${job.id}/config`;
  $('checkpoint-list').innerHTML=(job.checkpoints||[]).map(name=>`<a href="/api/jobs/${job.id}/checkpoints/${encodeURIComponent(name)}">${escapeHTML(name)} ↓</a>`).join('')||'<span class="muted">No periodic checkpoints yet</span>';
  $('chart').closest('.panel').hidden=evaluating;
  $('checkpoint-list').closest('.panel').hidden=evaluating;
  $('evaluation-panel').hidden=!evaluating||predicting;
  $('prediction-panel').hidden=!predicting;
  $('prediction-results').innerHTML=job.prediction?`<p>${job.prediction.samples} structures; energy in eV, forces in eV/angstrom. Forces are flattened; offsets identify each structure.</p>`+['energy.npy','forces.npy','offsets.npy','prediction'].map(f=>`<a href="/api/jobs/${job.id}/${f}">Download ${f==='prediction'?'metadata JSON':f}</a>`).join(''):'No completed predictions yet';
  if(predicting&&job.status==='running') $('detail-status').textContent='Predicting';
  $('evaluation-artifacts-panel').hidden=!evaluating||predicting;
  $('download-evaluation').hidden=!job.evaluation;
  $('download-evaluation').href=`/api/jobs/${job.id}/evaluation`;
  const plots=job.evaluation?.plots;
  const plotHTML=plots?Object.keys(plots).map(key=>{
    const url=`/api/jobs/${job.id}/plots/${encodeURIComponent(key)}`;
    return `<article class="evaluation-plot"><h3>${escapeHTML(key)}</h3><img src="${url}.png" alt="${escapeHTML(key)}: reference versus prediction" loading="lazy"><div><a href="${url}.png" download="${key}.png">Download PNG</a> · <a href="${url}.svg" download="${key}.svg">Download SVG</a></div></article>`;
  }).join(''):(job.evaluation?'<p class="footnote">No plots for this job. Run evaluation again to generate them.</p>':'');
  if($('evaluation-plots').innerHTML!==plotHTML) $('evaluation-plots').innerHTML=plotHTML;
  const artifacts=job.evaluation?.artifacts;
  const artifactFiles=artifacts?Object.keys(artifacts.files||{}):[];
  $('evaluation-artifacts').innerHTML=artifactFiles.length?'<strong>Prediction and reference arrays</strong>'+artifactFiles.map(name=>`<a href="/api/jobs/${job.id}/artifacts/${encodeURIComponent(name)}" download="${escapeHTML(name)}">${escapeHTML(name)} &#8595;</a>`).join('')+`<a href="/api/jobs/${job.id}/artifacts/${encodeURIComponent(artifacts.manifest)}" download="metadata.json">metadata.json &#8595;</a>`:(job.evaluation?'<span class="muted">No exported arrays</span>':'');
  $('evaluation-metrics').innerHTML=job.evaluation?'<table><thead><tr><th>Metric</th><th>MAE</th><th>RMSE</th><th>MSE</th></tr></thead><tbody>'+Object.entries(job.evaluation.metrics).map(([key,m])=>`<tr><td>${escapeHTML(key)}</td><td>${m.mae.toExponential(5)}</td><td>${m.rmse.toExponential(5)}</td><td>${m.mse.toExponential(5)}</td></tr>`).join('')+'</tbody></table>':'No complete evaluation results yet';
  drawChart();
}
function drawChart() {
  if(!current) return;
  const metric=text('metric'), history=current.history||[];
  const lastTest=history.filter(r=>Number.isFinite(r.test?.[metric])).at(-1);
  $('test-result').textContent=lastTest?`Latest test · Epoch ${lastTest.epoch} · ${metric} MSE = ${lastTest.test[metric].toExponential(5)}`:'No test results for this target yet';
  const values=history.flatMap(r=>[r.train?.[metric],r.validation?.[metric],r.test?.[metric]]).filter(Number.isFinite);
  if(!values.length){$('chart').innerHTML='<div class="empty"><p>Waiting for the first epoch of '+escapeHTML(metric)+' loss</p></div>';return;}
  const W=900,H=230,L=80,R=20,T=15,B=32;
  let lo=Math.min(...values),hi=Math.max(...values);
  if(hi===lo){const pad=Math.abs(hi)*.1||1;lo=Math.max(0,lo-pad);hi+=pad;}
  const x=e=>L+(e-1)/Math.max(1,history.length-1)*(W-L-R),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${metric} loss by epoch">`;
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,py=y(v);svg+=`<line x1="${L}" x2="${W-R}" y1="${py}" y2="${py}" stroke="#30333e"/><text x="${L-12}" y="${py+4}" text-anchor="end" fill="#959aa9" font-size="11">${v.toExponential(2)}</text>`;}
  for(const [kind,color] of [['train','#a18bff'],['validation','#6ad4ae'],['test','#f4bb75']]) {
    const points=history.filter(r=>Number.isFinite(r[kind]?.[metric]));
    svg+=`<polyline fill="none" stroke="${color}" stroke-width="2.5" points="${points.map(r=>`${x(r.epoch)},${y(r[kind][metric])}`).join(' ')}"/>`;
    for(const r of points) svg+=`<circle cx="${x(r.epoch)}" cy="${y(r[kind][metric])}" r="3" fill="${color}"><title>Epoch ${r.epoch}: ${r[kind][metric]}</title></circle>`;
  }
  svg+=`<text x="${L}" y="${H-6}" fill="#959aa9" font-size="11">Epoch 1</text><text x="${W-R}" y="${H-6}" text-anchor="end" fill="#959aa9" font-size="11">Epoch ${history.length}</text></svg>`;
  $('chart').innerHTML=svg;
}
async function refresh() {
  if(pollBusy) return; pollBusy=true;
  const hash=location.hash;
  try {
    jobs=await api('/api/jobs'); renderJobs(); $('connection').textContent='● Connected';
    if(hash.startsWith('#job/')) {
      const id=hash.slice(5),job=jobs.find(j=>j.id===id);
      if(!job) throw new Error('Job not found');
      const log=await api(`/api/jobs/${id}/log`);
      if(location.hash!==hash) return;
      renderDetail(job); $('log').textContent=log.text || 'Waiting for worker output...';
    }
  } catch(error) { $('connection').textContent='Connection or job retrieval failed'; notify(error.message); }
  finally { pollBusy=false; }
}
$('family').addEventListener('change',()=>{$('model-config').value=JSON.stringify(models[text('family')],null,2);updateBackendTargets();});
function updateBackendTargets() {
  const targets=backends[text('family')]?.training_targets||[];
  $('model-config-hint').textContent=backends[text('family')]?.config_hint||'Defaults work for new models; continuation requires the original architecture.';
  const evaluating=text('task-type')!=='training';
  for(const key of ['energy','forces','charges']) {
    const supported=targets.includes(key);
    $('eval-'+key).disabled=!supported;
    $('weight-'+key).disabled=evaluating||!supported||(key==='charges'&&!$('train-charges').checked);
  }
  $('train-charges').disabled=evaluating||!targets.includes('charges');
  $('charge-constraint').disabled=!targets.includes('charges');
  if(!targets.includes('charges')) $('charge-constraint').checked=false;
}
function updateTaskType() {
  const predicting=text('task-type')==='prediction';
  for(const option of $('family').options) option.disabled=predicting?option.value!=='uma':option.value==='uma';
  const family=predicting?'uma':text('family')==='uma'?'newtonnet':text('family');
  if(text('family')!==family){$('family').value=family;$('model-config').value=JSON.stringify(models[family],null,2);}
  if(predicting){$('file-charge').value='';$('file-spin').value='';$('dtype').value='float32';$('model-config').closest('details').open=true;}
  $('dtype').disabled=predicting;
  $('file-spin').closest('label').hidden=!predicting;
  for(const id of ['file-energy','file-forces','file-charges','energy-scale','gradients']) $(id).closest('label').hidden=predicting;
  $('file-charge').closest('label').querySelector('small').textContent=predicting?'Optional total charge per structure; leave blank to use the model settings.':'Required for the hard constraint: charge.npy with shape [structures] or [structures, 1], in e.';
  const evaluating=text('task-type')!=='training';
  updateTrainingMode();
  $('split-hint').hidden=evaluating;
  $('training-hint').hidden=evaluating;
  $('start').textContent=evaluating?'Submit evaluation →':'Submit training →';
  for(const id of ['epochs','batch-size','learning-rate','seed','weight-energy','weight-forces','train-charges','weight-charges','save-interval','max-checkpoints','test-interval','validation-shards','validation-directory','test-shards','test-directory']) {
    $(id).closest('label').hidden=evaluating; $(id).disabled=evaluating;
  }
  for(const id of ['eval-energy','eval-forces','eval-charges']) $(id).closest('label').hidden=!evaluating;
  for(const split of ['validation','test']) $('managed-'+split).closest('label').hidden=evaluating;
  for(const id of ['early-stopping','early-monitor','early-patience','early-min-delta']) $(id).closest('label').hidden=evaluating;
  updateEarlyStopping();
  updateBackendTargets();
  $('new-view').querySelector('h1').textContent=evaluating?'Create evaluation job':'Create training job';
  $('new-view').querySelector('.page-title p').textContent=evaluating?'Select a checkpoint and labeled NPY data to calculate errors. Configuration must match the model.':'Select a model, connect NumPy data and start training.';
  $('preview-result').textContent='Check files, array shapes and labels.';
  for(const id of ['eval-energy','eval-forces','eval-charges','charge-constraint']) if(predicting) $(id).closest('label').hidden=true;
  if(!predicting) $('charge-constraint').closest('label').hidden=false;
  if(predicting){
    $('start').textContent='Submit prediction →';
    $('new-view').querySelector('h1').textContent='Create UMA prediction job';
    $('new-view').querySelector('.page-title p').textContent='Predict energy and forces from NPY structures. Reference labels are not required.';
    $('checkpoint-hint').textContent='Required for prediction';
  }
}
$('task-type').addEventListener('change',updateTaskType);
function updateTrainingMode(){
  const evaluating=text('task-type')!=='training';
  const needsCheckpoint=evaluating||text('training-mode')==='continue';
  $('training-mode').closest('label').hidden=evaluating;
  $('training-mode').disabled=evaluating;
  $('checkpoint').closest('label').hidden=!needsCheckpoint;
  $('checkpoint').disabled=!needsCheckpoint;
  $('checkpoint').required=needsCheckpoint;
  $('checkpoint-hint').textContent=evaluating?'Required for evaluation':'Required for continuation';
}
$('training-mode').addEventListener('change',updateTrainingMode);
function updateEarlyStopping(){
  const evaluating=text('task-type')!=='training';
  $('early-stopping').disabled=evaluating;
  for(const id of ['early-monitor','early-patience','early-min-delta']) $(id).disabled=evaluating||!$('early-stopping').checked;
}
$('early-stopping').addEventListener('change',updateEarlyStopping);
$('train-charges').addEventListener('change',updateBackendTargets);
$('layout').addEventListener('change',()=>{
  const standard=text('layout')==='standard';
  if(text('layout')==='custom'){$('file-details').open=true;return;}
  const mapping=standard?{z:'z.npy',pos:'pos.npy',energy:'energy.npy',forces:'forces.npy',charges:'charges.npy',charge:'charge.npy'}:{z:'full_qm_type.npy',pos:'qm_coord_{shard}.npy',energy:'energy_{shard}.npy',forces:'qm_grad_{shard}.npy',charges:'qm_charge_{shard}.npy',charge:'total_charge_{shard}.npy'};
  Object.entries(mapping).forEach(([k,v])=>$('file-'+k).value=v);
  if(text('task-type')==='prediction') $('file-charge').value='';
  $('shards').value=standard?'':'w00, w01'; $('validation-shards').value=''; $('test-shards').value=''; $('gradients').checked=!standard;
});
$('preview').addEventListener('click',async()=>{
  notify(); $('preview').disabled=true; $('preview-result').textContent='Checking...';
  try { const data=await api('/api/preview',payload()); const main=data.prediction||data.evaluation||data.train; $('preview-result').textContent=`✓ ${main.samples}  ${data.prediction?'prediction':data.evaluation?'evaluation':'training'} structures / ${main.groups} groups`+(data.validation?` · ${data.validation.samples} validation structures`:'')+(data.test?` · ${data.test.samples} test structures`:''); }
  catch(error){notify(error.message);$('preview-result').textContent='Check failed. Review the dataset settings.';}
  finally{$('preview').disabled=false;}
});
$('training-form').addEventListener('submit',async event=>{
  event.preventDefault(); notify(); $('start').disabled=true;
  try {const job=await api('/api/jobs',payload());location.hash='#job/'+job.id;}
  catch(error){notify(error.message);}
  finally{$('start').disabled=false;}
});
$('stop').addEventListener('click',async()=>{
  if(!current)return; $('stop').disabled=true;
  try{renderDetail(await api(`/api/jobs/${current.id}/stop`,{}));}
  catch(error){notify(error.message);$('stop').disabled=false;}
});
$('metric').addEventListener('change',drawChart);
window.addEventListener('hashchange',route);
async function init(){
  $('start').disabled=true;
  try{const config=await api('/api/presets');models=config.models;backends=config.backends;
    $('family').replaceChildren(...Object.keys(models).map(name=>new Option(backendLabel(name),name)));
    $('model-config').value=JSON.stringify(models[text('family')],null,2);$('runs-root').textContent='Output location · '+config.root;
    const gpu=$('device').querySelector('[value="cuda"]');gpu.disabled=!config.cuda;if(!config.cuda)gpu.textContent='GPU · CUDA not detected';$('start').disabled=false;
  }catch(error){notify(error.message);}
  updateTaskType();route();setInterval(refresh,2000);
}
