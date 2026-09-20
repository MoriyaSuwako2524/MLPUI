const $ = id => document.getElementById(id);
const active = new Set(['starting','running','stopping']);
const statuses = {starting:'准备中',running:'训练中',completed:'已完成',stopped:'已停止',failed:'失败',interrupted:'已中断'};
let jobs = [], models = {}, current = null, pollBusy = false;
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function notify(message='') { $('notice').textContent=message; $('notice').hidden=!message; }
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {cache:'no-store'} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const result = await response.json();
  if(!response.ok) throw new Error(result.error || '请求失败');
  return result;
}
function text(id) { return $(id).value.trim(); }
function number(id) { const v=Number(text(id)); if(!text(id)||!Number.isFinite(v)) throw new Error('请填写有效的数值'); return v; }
function groups(id) { return text(id).split(',').map(v=>v.trim()).filter(Boolean); }
function payload() {
  const evaluating=text('task-type')==='evaluation';
  const weights={};
  for(const key of ['energy','forces','charges']) {
    if(evaluating) {if($('eval-'+key).checked) weights[key]=1;}
    else if((key!=='charges'||$('train-charges').checked)&&number('weight-'+key)>0) weights[key]=number('weight-'+key);
  }
  if(!Object.keys(weights).length) throw new Error('至少选择能量、力或原子电荷中的一项');
  const files={};
  for(const key of ['z','pos','energy','forces','charges','cell','pbc','offsets']) {
    if(['energy','forces','charges'].includes(key) && !(key in weights)) continue;
    if(text('file-'+key)) files[key]=text('file-'+key);
  }
  for(const key of Object.keys(weights)) if(!files[key]) throw new Error('请填写 '+key+' 标签文件');
  const train={directory:text('directory'),files,gradients:$('gradients').checked,energy_scale:number('energy-scale'),length_scale:number('length-scale')};
  if(groups('shards').length) train.shards=groups('shards');
  let model;
  try { model=JSON.parse($('model-config').value); } catch { throw new Error('模型结构配置不是有效的 JSON'); }
  const modelSettings=text('family')==='newtonnet'?(model.model||model):model;
  if($('charge-constraint').checked) {
    if(!('charges' in weights)) throw new Error('请同时启用原子电荷训练或评估');
    modelSettings.charge_constraint=true;
  }
  if(modelSettings.charge_constraint) {
    if(!text('file-charge')) throw new Error('硬约束需要每个结构的总电荷 Q 文件');
    files.charge=text('file-charge');
  }
  if(evaluating) {
    if(!text('checkpoint')) throw new Error('评估需要填写已有模型路径');
    return {task_type:'evaluation',name:text('name'),family:text('family'),model_config:model,
      checkpoint:text('checkpoint'),evaluation:train,
      training:{dtype:text('dtype'),device:text('device'),loss_weights:weights}};
  }
  const result={name:text('name'),family:text('family'),model_config:model,train,
    training:{epochs:number('epochs'),batch_size:number('batch-size'),learning_rate:number('learning-rate'),dtype:text('dtype'),device:text('device'),seed:number('seed'),loss_weights:weights,save_interval:number('save-interval'),max_checkpoints:number('max-checkpoints'),test_interval:number('test-interval')}};
  if(text('checkpoint')) result.checkpoint=text('checkpoint');
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
  return result;
}
function route() {
  const hash=location.hash || '#jobs';
  const view=hash==='#new'?'new':hash.startsWith('#job/')?'detail':'jobs';
  for(const name of ['jobs','new','detail']) $(name+'-view').hidden=name!==view;
  $('nav-jobs').classList.toggle('active',view!=='new'); $('nav-new').classList.toggle('active',view==='new');
  $('breadcrumb').textContent=view==='new'?'新建任务':view==='detail'?'任务详情':'任务列表';
  notify(); current=null;
  if(view==='detail') {
    $('detail-name').textContent='加载中…'; $('detail-meta').textContent=''; $('log').textContent='加载中…';
    $('stop').hidden=true; $('download-model').hidden=true;
  }
  refresh();
}
function renderJobs() {
  $('count-all').textContent=jobs.length;
  $('count-active').textContent=jobs.filter(j=>active.has(j.status)).length;
  $('count-done').textContent=jobs.filter(j=>j.status==='completed').length;
  if(!jobs.length) {
    $('job-list').innerHTML='<div class="empty"><div class="empty-icon">▦</div><h2>开始你的第一个任务</h2><p>连接 .npy 数据，配置模型。每次实验都有清晰的记录。</p><a class="button primary" href="#new">＋ 创建任务</a></div>';
    return;
  }
  $('job-list').innerHTML='<div class="table-wrap"><table><thead><tr><th>任务</th><th>模型</th><th>状态</th><th>进度</th><th>创建时间</th></tr></thead><tbody>'+jobs.map(j=>`<tr><td><a href="#job/${j.id}">${escapeHTML(j.name)}</a><small>${j.task_type==='evaluation'?'评估':'训练'} · ${j.id.slice(0,8)}</small></td><td>${j.family==='newtonnet'?'NewtonNet':'TorchMD-Net'}</td><td><span class="badge ${escapeHTML(j.status)}">${j.stop_requested&&active.has(j.status)?'正在停止':(j.task_type==='evaluation'&&j.status==='running'?'评估中':statuses[j.status])||escapeHTML(j.status)}</span></td><td>${j.task_type==='evaluation'?`${j.completed||0} / ${j.summary.evaluation.samples} 结构`:`${j.history.length} / ${j.epochs} epochs`}</td><td>${escapeHTML(new Date(j.created*1000).toLocaleString())}</td></tr>`).join('')+'</tbody></table></div>';
}
function renderDetail(job) {
  current=job;
  const evaluating=job.task_type==='evaluation';
  $('detail-name').textContent=job.name;
  $('detail-meta').textContent=`${job.family==='newtonnet'?'NewtonNet':'TorchMD-Net'} · ${(job.summary.evaluation||job.summary.train).samples} 个${evaluating?'评估':'训练'}结构 · ${job.id.slice(0,8)}`;
  $('detail-status').className='badge '+job.status;
  $('detail-status').textContent=job.stop_requested&&active.has(job.status)?'正在停止':statuses[job.status];
  const partial=job.phase==='training'?(job.completed||0)/(job.total||1):['validation','test'].includes(job.phase)?1:0;
  const epoch=job.history.length;
  $('progress').value=Math.min(100,100*(epoch+partial)/job.epochs);
  $('progress-label').textContent=`${epoch} / ${job.epochs} epochs`+(job.phase==='training'?` · ${job.completed} / ${job.total} 结构`:job.phase==='loading'?' · 正在加载模型与数据':job.phase==='validation'?' · 正在验证':job.phase==='test'?' · 正在评估测试集':'');
  if(evaluating) {
    $('progress').value=100*(job.completed||0)/(job.summary.evaluation.samples||1);
    $('progress-label').textContent=`${job.completed||0} / ${job.summary.evaluation.samples} 结构`+(job.phase==='loading'?' · 正在加载模型与数据':'');
    if(job.status==='running') $('detail-status').textContent='评估中';
  }
  $('detail-error').hidden=!job.error; $('detail-error').textContent=job.error||'';
  $('stop').hidden=!active.has(job.status); $('stop').disabled=job.stop_requested;
  $('stop').textContent=job.stop_requested?'正在停止…':'停止任务';
  $('output-path').textContent=job.directory;
  $('download-model').hidden=!job.checkpoint || !['completed','stopped'].includes(job.status);
  $('download-model').href=`/api/jobs/${job.id}/model`;
  $('download-config').href=`/api/jobs/${job.id}/config`;
  $('checkpoint-list').innerHTML=(job.checkpoints||[]).map(name=>`<a href="/api/jobs/${job.id}/checkpoints/${encodeURIComponent(name)}">${escapeHTML(name)} ↓</a>`).join('')||'<span class="muted">尚无定期保存的 checkpoint</span>';
  $('chart').closest('.panel').hidden=evaluating;
  $('checkpoint-list').closest('.panel').hidden=evaluating;
  $('evaluation-panel').hidden=!evaluating;
  $('download-evaluation').hidden=!job.evaluation;
  $('download-evaluation').href=`/api/jobs/${job.id}/evaluation`;
  $('evaluation-metrics').innerHTML=job.evaluation?'<table><thead><tr><th>指标</th><th>MAE</th><th>RMSE</th><th>MSE</th></tr></thead><tbody>'+Object.entries(job.evaluation.metrics).map(([key,m])=>`<tr><td>${escapeHTML(key)}</td><td>${m.mae.toExponential(5)}</td><td>${m.rmse.toExponential(5)}</td><td>${m.mse.toExponential(5)}</td></tr>`).join('')+'</tbody></table>':'尚无完整评估结果';
  drawChart();
}
function drawChart() {
  if(!current) return;
  const metric=text('metric'), history=current.history;
  const lastTest=history.filter(r=>Number.isFinite(r.test?.[metric])).at(-1);
  $('test-result').textContent=lastTest?`最近测试 · Epoch ${lastTest.epoch} · ${metric} MSE = ${lastTest.test[metric].toExponential(5)}`:'尚无该项测试集评估记录';
  const values=history.flatMap(r=>[r.train?.[metric],r.validation?.[metric],r.test?.[metric]]).filter(Number.isFinite);
  if(!values.length){$('chart').innerHTML='<div class="empty"><p>等待首个 epoch 的 '+escapeHTML(metric)+' 损失记录</p></div>';return;}
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
    jobs=await api('/api/jobs'); renderJobs(); $('connection').textContent='● 已连接';
    if(hash.startsWith('#job/')) {
      const id=hash.slice(5),job=jobs.find(j=>j.id===id);
      if(!job) throw new Error('未找到该训练任务');
      const log=await api(`/api/jobs/${id}/log`);
      if(location.hash!==hash) return;
      renderDetail(job); $('log').textContent=log.text || '等待训练进程输出…';
    }
  } catch(error) { $('connection').textContent='连接或任务读取异常'; notify(error.message); }
  finally { pollBusy=false; }
}
$('family').addEventListener('change',()=>{$('model-config').value=JSON.stringify(models[text('family')],null,2);});
function updateTaskType() {
  const evaluating=text('task-type')==='evaluation';
  $('checkpoint').required=evaluating;
  $('split-hint').hidden=evaluating;
  $('training-hint').hidden=evaluating;
  $('start').textContent=evaluating?'开始评估 →':'开始训练 →';
  for(const id of ['epochs','batch-size','learning-rate','seed','weight-energy','weight-forces','train-charges','weight-charges','save-interval','max-checkpoints','test-interval','validation-shards','validation-directory','test-shards','test-directory']) {
    $(id).closest('label').hidden=evaluating; $(id).disabled=evaluating;
  }
  for(const id of ['eval-energy','eval-forces','eval-charges']) $(id).closest('label').hidden=!evaluating;
  $('weight-charges').disabled=evaluating||!$('train-charges').checked;
  $('new-view').querySelector('h1').textContent=evaluating?'创建评估任务':'创建训练任务';
  $('new-view').querySelector('.page-title p').textContent=evaluating?'选择已有模型和带标签的 npy 数据，计算误差。模型结构配置须与原模型一致。':'选择模型，连接 NumPy 数据，然后开始训练。';
  $('preview-result').textContent='检查文件、数组形状与标签。';
}
$('task-type').addEventListener('change',updateTaskType);
$('train-charges').addEventListener('change',()=>{$('weight-charges').disabled=!$('train-charges').checked;});
$('layout').addEventListener('change',()=>{
  const standard=text('layout')==='standard';
  if(text('layout')==='custom'){$('file-details').open=true;return;}
  const mapping=standard?{z:'z.npy',pos:'pos.npy',energy:'energy.npy',forces:'forces.npy',charges:'charges.npy',charge:'charge.npy'}:{z:'full_qm_type.npy',pos:'qm_coord_{shard}.npy',energy:'energy_{shard}.npy',forces:'qm_grad_{shard}.npy',charges:'qm_charge_{shard}.npy',charge:'total_charge_{shard}.npy'};
  Object.entries(mapping).forEach(([k,v])=>$('file-'+k).value=v);
  $('shards').value=standard?'':'w00, w01'; $('validation-shards').value=''; $('test-shards').value=''; $('gradients').checked=!standard;
});
$('preview').addEventListener('click',async()=>{
  notify(); $('preview').disabled=true; $('preview-result').textContent='正在检查…';
  try { const data=await api('/api/preview',payload()); const main=data.evaluation||data.train; $('preview-result').textContent=`✓ ${main.samples} 个${data.evaluation?'评估':'训练'}结构 / ${main.groups} 组`+(data.validation?` · ${data.validation.samples} 个验证结构`:'')+(data.test?` · ${data.test.samples} 个测试结构`:''); }
  catch(error){notify(error.message);$('preview-result').textContent='检查未通过，请核对数据配置。';}
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
  try{const config=await api('/api/presets');models=config.models;$('model-config').value=JSON.stringify(models.newtonnet,null,2);$('runs-root').textContent='输出位置 · '+config.root;
    const gpu=$('device').querySelector('[value="cuda"]');gpu.disabled=!config.cuda;if(!config.cuda)gpu.textContent='GPU · 未检测到 CUDA';$('start').disabled=false;
  }catch(error){notify(error.message);}
  updateTaskType();route();setInterval(refresh,2000);
}
init();
