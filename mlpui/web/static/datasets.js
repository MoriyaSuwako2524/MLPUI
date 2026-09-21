let datasetRecords=[], selectedDataset=null, datasetBusy=false, uploadDraft=null;
const readyDatasets=()=>datasetRecords.filter(d=>d.status==='ready'&&!d.archived);
async function loadDatasets(){
  datasetRecords=await api('/api/datasets');
  const previousTag=text('dataset-tag-filter');
  const tags=[...new Set(datasetRecords.flatMap(d=>d.tags||[]))].sort((a,b)=>a.localeCompare(b));
  $('dataset-tag-filter').innerHTML='<option value="">全部标签</option><option value="__untagged__">未分类</option>'+tags.map(t=>`<option value="tag:${escapeHTML(t)}">${escapeHTML(t)}</option>`).join('');
  if([...$('dataset-tag-filter').options].some(o=>o.value===previousTag))$('dataset-tag-filter').value=previousTag;
  for(const split of ['train','validation','test']){
    const select=$('managed-'+split), previous=select.value;
    select.innerHTML=`<option value="">${split==='train'?'手动填写路径和文件映射':'使用下方手动配置 / 不使用'}</option>`+
      readyDatasets().map(d=>`<option value="${d.id}">${escapeHTML(d.name)} · ${d.summary.samples} 个结构</option>`).join('');
    if(readyDatasets().some(d=>d.id===previous)) select.value=previous;
  }
  setManagedFields(); renderDatasets();
}
function setManagedFields(){
  const managed=!!text('managed-train');
  const selected=datasetRecords.find(d=>d.id===text('managed-train'));
  if(selected){
    const spec=selected.spec;
    $('directory').value=spec.directory; $('layout').value='custom'; $('shards').value=(spec.shards||[]).join(', ');
    $('gradients').checked=!!spec.gradients;
    $('energy-scale').value=spec.energy_scale??1; $('length-scale').value=spec.length_scale??1;
    for(const input of $('file-details').querySelectorAll('input[id^="file-"]')){
      const key=input.id.slice(5);input.value=spec.files?.[key]||(selected.summary.fields.includes(key)?key+'.npy':'');
    }
  }
  for(const id of ['directory','layout','shards','gradients','energy-scale','length-scale']) $(id).disabled=managed;
  for(const input of $('file-details').querySelectorAll('input')) input.disabled=managed;
  for(const split of ['validation','test']){
    const child=datasetRecords.find(d=>d.id===text('managed-'+split));
    if(child){$(split+'-directory').value=child.spec.directory;$(split+'-shards').value=(child.spec.shards||[]).join(', ');}
    const disabled=text('task-type')==='evaluation'||!!text('managed-'+split);
    $(split+'-directory').disabled=disabled; $(split+'-shards').disabled=disabled;
  }
}
function renderDatasets(){
  const query=text('dataset-search').toLowerCase();
  const tag=text('dataset-tag-filter');
  const records=datasetRecords.filter(d=>($('dataset-show-archived').checked||!d.archived)&&
    (!tag||(tag==='__untagged__'?!(d.tags||[]).length:(d.tags||[]).includes(tag.slice(4))))&&
    `${d.name} ${(d.tags||[]).join(' ')} ${d.spec?.directory||''} ${d.summary?.fields.join(' ')||''}`.toLowerCase().includes(query));
  $('dataset-list').innerHTML=records.length?'<div class="dataset-cards">'+records.map(d=>`<article class="panel dataset-card">
    <div class="dataset-card-title"><h2>${escapeHTML(d.name)}</h2><span class="badge ${d.status==='ready'?'completed':''}">${d.archived?'已归档':d.status==='ready'?'可用':'上传草稿'}</span></div>
    <strong>${d.summary?`${d.summary.samples.toLocaleString()} 个结构`:'等待上传和检查'}</strong>
    <div class="dataset-tags">${(d.tags||[]).map(t=>`<span class="badge">${escapeHTML(t)}</span>`).join(' ')||'<span class="muted">未分类</span>'}</div>
    <p>${escapeHTML(d.summary?.fields.join(' · ')||'NPY')}</p><p class="footnote">${escapeHTML(d.spec?.directory||'本机上传')}</p>
    <div class="preview-row"><button data-dataset-action="detail" data-id="${d.id}">管理</button>${d.status==='ready'&&!d.archived?`<button data-dataset-action="use" data-id="${d.id}">用于任务 →</button>`:''}</div></article>`).join('')+'</div>':'<div class="panel empty"><h2>暂无匹配的数据集</h2><p>添加服务器目录或上传本机的 npy 文件。</p></div>';
}
function showDataset(id){
  selectedDataset=datasetRecords.find(d=>d.id===id);
  const d=selectedDataset;
  $('dataset-detail').hidden=false;
  $('dataset-detail-title').textContent=d.name; $('dataset-rename').value=d.name;
  $('dataset-tags-edit').value=(d.tags||[]).join(', ');
  $('dataset-delete-panel').open=false;$('dataset-delete-files').checked=false;
  $('dataset-delete-files').disabled=d.source==='existing';
  $('dataset-delete-files').closest('label').hidden=d.source==='existing';
  $('dataset-delete-info').textContent=`将删除「${d.name}」${d.source==='existing'?'的登记记录；服务器源文件始终保留。':'。可选择仅移除记录，或连同托管文件一起删除。'}`;
  $('dataset-detail-info').textContent=`${d.spec?.directory||'上传草稿'}${d.parent_id?' · 源数据集 '+d.parent_id:''}`;
  $('dataset-archive').textContent=d.archived?'恢复到列表':'归档';
  $('dataset-inspect').disabled=d.status!=='ready';
  $('dataset-split-details').hidden=d.status!=='ready'||d.archived;
  $('dataset-split-name').value=d.name.slice(0,65)+'-split';
  $('dataset-file-list').innerHTML=d.summary?'<table><thead><tr><th>字段</th><th>分组 / 文件</th><th>形状</th><th>类型</th><th>大小</th></tr></thead><tbody>'+d.summary.files.map(f=>`<tr><td>${escapeHTML(f.field)}</td><td>${escapeHTML(f.group)}<small>${escapeHTML(f.filename)}</small></td><td>${escapeHTML(f.shape.join(' × '))}</td><td>${escapeHTML(f.dtype)}</td><td>${(f.bytes/1048576).toFixed(2)} MiB</td></tr>`).join('')+'</tbody></table>':'';
}
function useDataset(id){
  $('managed-train').value=id;
  const d=datasetRecords.find(d=>d.id===id);
  for(const split of ['validation','test']){
    const sibling=d.split_id&&d.split_role==='train'?readyDatasets().find(v=>v.split_id===d.split_id&&v.split_role===split):null;
    $('managed-'+split).value=sibling?.id||'';
    $(split+'-directory').value=''; $(split+'-shards').value='';
  }
  setManagedFields();
}
function datasetSpec(){
  let mapping; try{mapping=JSON.parse(text('dataset-mapping')||'{}');}catch{throw new Error('文件映射不是有效 JSON');}
  if(!mapping||Array.isArray(mapping)||typeof mapping!=='object')throw new Error('文件映射应为 JSON 对象');
  const spec={directory:text('dataset-directory'),gradients:$('dataset-gradients').checked,
    energy_scale:number('dataset-energy-scale'),length_scale:number('dataset-length-scale')};
  if(Object.keys(mapping).length)spec.files=mapping;
  if(groups('dataset-shards').length)spec.shards=groups('dataset-shards');
  return spec;
}
async function datasetOperation(action){
  if(datasetBusy)return;
  datasetBusy=true; notify();
  const controls=[...$('datasets-view').querySelectorAll('button')]; controls.forEach(b=>b.disabled=true);
  try{await action();await loadDatasets();if(selectedDataset)showDataset(selectedDataset.id);}
  catch(error){notify(error.message);$('dataset-progress').textContent='操作未完成，请检查提示后重试。';$('dataset-split-progress').textContent='';}
  finally{datasetBusy=false;controls.forEach(b=>b.disabled=false);if(selectedDataset)$('dataset-inspect').disabled=selectedDataset.status!=='ready';}
}
$('dataset-new').addEventListener('click',()=>{$('dataset-create-panel').hidden=false;$('dataset-create-panel').scrollIntoView({behavior:'smooth',block:'start'});});
$('dataset-cancel').addEventListener('click',()=>{$('dataset-create-panel').hidden=true;});
$('dataset-source').addEventListener('change',()=>{
  const uploading=text('dataset-source')==='upload';
  $('dataset-directory-label').hidden=uploading; $('dataset-directory').required=!uploading;
  $('dataset-upload-label').hidden=!uploading; uploadDraft=null;
});
$('dataset-layout').addEventListener('change',()=>{
  const qm=text('dataset-layout')==='qm';
  if(text('dataset-layout')==='custom'){$('dataset-mapping-details').open=true;return;}
  $('dataset-mapping').value=qm?JSON.stringify({z:'full_qm_type.npy',pos:'qm_coord_{shard}.npy',energy:'energy_{shard}.npy',forces:'qm_grad_{shard}.npy'},null,2):'{}';
  $('dataset-shards').value=qm?'w00, w01':''; $('dataset-gradients').checked=qm;
});
$('dataset-form').addEventListener('submit',event=>{
  event.preventDefault();datasetOperation(async()=>{
    const spec=datasetSpec(),name=text('dataset-name'),tags=parseDatasetTags('dataset-tags-new');let record;
    if(text('dataset-source')==='upload'){
      const files=[...$('dataset-files').files];
      if(!files.length)throw new Error('请选择 npy 文件');
      if(files.some(f=>!f.name.endsWith('.npy')))throw new Error('只支持 .npy 文件');
      // Retain the draft after a failed validation so mapping can be corrected.
      const signature=JSON.stringify(files.map(f=>[f.name,f.size,f.lastModified]));
      if(!uploadDraft||uploadDraft.signature!==signature){
        record=await api('/api/datasets',{name,tags});uploadDraft={id:record.id,signature,uploaded:new Set()};
      }
      for(let i=0;i<files.length;i++){
        const file=files[i]; if(uploadDraft.uploaded.has(file.name))continue;
        $('dataset-progress').textContent=`正在上传 ${i+1}/${files.length}：${file.name}`;
        const response=await fetch(`/api/datasets/${uploadDraft.id}/files/${encodeURIComponent(file.name)}`,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});
        const result=await response.json();if(!response.ok)throw new Error(result.error||'上传失败');
        uploadDraft.uploaded.add(file.name);
      }
      $('dataset-progress').textContent='正在检查数组与字段…';
      record=await api(`/api/datasets/${uploadDraft.id}/finalize`,spec);uploadDraft=null;
    }else{
      $('dataset-progress').textContent='正在检查数据…';
      record=await api('/api/datasets',{name,spec,tags});
    }
    $('dataset-progress').textContent=`已添加 ${record.summary.samples} 个结构`;
    selectedDataset=record;
  });
});
$('dataset-list').addEventListener('click',event=>{
  const button=event.target.closest('[data-dataset-action]');if(!button||datasetBusy)return;
  if(button.dataset.datasetAction==='use'){useDataset(button.dataset.id);location.hash='#new';}
  else {showDataset(button.dataset.id);$('dataset-detail').scrollIntoView({behavior:'smooth',block:'start'});}
});
$('dataset-search').addEventListener('input',renderDatasets);
$('dataset-show-archived').addEventListener('change',renderDatasets);
$('dataset-tag-filter').addEventListener('change',renderDatasets);
function parseDatasetTags(id){return text(id).split(/[,，]/).map(v=>v.trim()).filter(Boolean);}
$('dataset-tags-save').addEventListener('click',()=>datasetOperation(async()=>{
  await api(`/api/datasets/${selectedDataset.id}/update`,{tags:parseDatasetTags('dataset-tags-edit')});
}));
$('dataset-delete-confirm').addEventListener('click',()=>datasetOperation(async()=>{
  const id=selectedDataset.id;
  await api(`/api/datasets/${id}/delete`,{delete_files:$('dataset-delete-files').checked});
  if(uploadDraft?.id===id)uploadDraft=null;
  selectedDataset=null;$('dataset-detail').hidden=true;
  for(const split of ['train','validation','test']) if(text('managed-'+split)===id){
    $('managed-'+split).value='';
    $(split==='train'?'directory':split+'-directory').value='';
    $(split==='train'?'shards':split+'-shards').value='';
  }
}));
$('dataset-rename-save').addEventListener('click',()=>datasetOperation(async()=>{
  await api(`/api/datasets/${selectedDataset.id}/update`,{name:text('dataset-rename')});
}));
$('dataset-archive').addEventListener('click',()=>datasetOperation(async()=>{
  await api(`/api/datasets/${selectedDataset.id}/update`,{archived:!selectedDataset.archived});
}));
$('dataset-inspect').addEventListener('click',()=>datasetOperation(async()=>{
  const summary=await api(`/api/datasets/${selectedDataset.id}/inspect`,{});
  $('dataset-split-progress').textContent=`检查通过：${summary.samples} 个结构`;
}));
$('dataset-split-form').addEventListener('submit',event=>{
  event.preventDefault();datasetOperation(async()=>{
    $('dataset-split-progress').textContent='正在生成独立子集；大数据集可能需要较长时间…';
    const result=await api(`/api/datasets/${selectedDataset.id}/split`,{name:text('dataset-split-name'),
      ratios:['train','validation','test'].map(k=>number('dataset-'+k+'-ratio')),
      seed:number('dataset-split-seed'),method:text('dataset-split-method')});
    $('dataset-split-progress').textContent='已生成：'+result.datasets.map(d=>`${d.split_role} ${d.summary.samples}`).join(' / ');
  });
});
$('managed-train').addEventListener('change',()=>{if(text('managed-train'))useDataset(text('managed-train'));else setManagedFields();});
for(const split of ['validation','test']) $('managed-'+split).addEventListener('change',()=>{
  if(!text('managed-'+split)){$(split+'-directory').value='';$(split+'-shards').value='';}
  setManagedFields();
});
$('task-type').addEventListener('change',setManagedFields);
init();
