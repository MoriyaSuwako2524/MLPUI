let datasetRecords=[], selectedDataset=null, datasetBusy=false, uploadDraft=null;
const readyDatasets=()=>datasetRecords.filter(d=>d.status==='ready'&&!d.archived);
async function loadDatasets(){
  datasetRecords=await api('/api/datasets');
  const previousTag=text('dataset-tag-filter');
  const tags=[...new Set(datasetRecords.flatMap(d=>d.tags||[]))].sort((a,b)=>a.localeCompare(b));
  $('dataset-tag-filter').innerHTML='<option value="">All tags</option><option value="__untagged__">Untagged</option>'+tags.map(t=>`<option value="tag:${escapeHTML(t)}">${escapeHTML(t)}</option>`).join('');
  if([...$('dataset-tag-filter').options].some(o=>o.value===previousTag))$('dataset-tag-filter').value=previousTag;
  for(const split of ['train','validation','test']){
    const select=$('managed-'+split), previous=select.value;
    select.innerHTML=`<option value="">${split==='train'?'Enter paths and file mapping manually':'Use manual settings below / None'}</option>`+
      readyDatasets().map(d=>`<option value="${d.id}">${escapeHTML(d.name)} · ${d.summary.samples} structures</option>`).join('');
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
    <div class="dataset-card-title"><h2>${escapeHTML(d.name)}</h2><span class="badge ${d.status==='ready'?'completed':''}">${d.archived?'Archived':d.status==='ready'?'Ready':'Upload draft'}</span></div>
    <strong>${d.summary?`${d.summary.samples.toLocaleString('en-US')} structures`:'Waiting for upload and validation'}</strong>
    <div class="dataset-tags">${(d.tags||[]).map(t=>`<span class="badge">${escapeHTML(t)}</span>`).join(' ')||'<span class="muted">Untagged</span>'}</div>
    <p>${escapeHTML(d.summary?.fields.join(' · ')||'NPY')}</p><p class="footnote">${escapeHTML(d.spec?.directory||'Local upload')}</p>
    <div class="preview-row"><button data-dataset-action="detail" data-id="${d.id}">Manage</button>${d.status==='ready'&&!d.archived?`<button data-dataset-action="use" data-id="${d.id}">Use in a job →</button>`:''}</div></article>`).join('')+'</div>':'<div class="panel empty"><h2>No matching datasets</h2><p>Add a server directory or upload local NPY files.</p></div>';
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
  $('dataset-delete-info').textContent=`Remove "${d.name}"${d.source==='existing'?' from the library; source files on the server are always retained.':'. Remove only the record, or delete its managed files as well.'}`;
  $('dataset-detail-info').textContent=`${d.spec?.directory||'Upload draft'}${d.parent_id?' · Source dataset '+d.parent_id:''}`;
  $('dataset-archive').textContent=d.archived?'Restore to list':'Archive';
  $('dataset-inspect').disabled=d.status!=='ready';
  $('dataset-split-details').hidden=d.status!=='ready'||d.archived;
  $('dataset-split-name').value=d.name.slice(0,65)+'-split';
  $('dataset-file-list').innerHTML=d.summary?'<table><thead><tr><th>Field</th><th>Group / file</th><th>Shape</th><th>Type</th><th>Size</th></tr></thead><tbody>'+d.summary.files.map(f=>`<tr><td>${escapeHTML(f.field)}</td><td>${escapeHTML(f.group)}<small>${escapeHTML(f.filename)}</small></td><td>${escapeHTML(f.shape.join(' × '))}</td><td>${escapeHTML(f.dtype)}</td><td>${(f.bytes/1048576).toFixed(2)} MiB</td></tr>`).join('')+'</tbody></table>':'';
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
  let mapping; try{mapping=JSON.parse(text('dataset-mapping')||'{}');}catch{throw new Error('File mapping is not valid JSON');}
  if(!mapping||Array.isArray(mapping)||typeof mapping!=='object')throw new Error('File mapping must be a JSON object');
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
  catch(error){notify(error.message);$('dataset-progress').textContent='Operation did not complete. Check the message and try again.';$('dataset-split-progress').textContent='';}
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
      if(!files.length)throw new Error('Select NPY files');
      if(files.some(f=>!f.name.endsWith('.npy')))throw new Error('Only .npy files are supported');
      // Retain the draft after a failed validation so mapping can be corrected.
      const signature=JSON.stringify(files.map(f=>[f.name,f.size,f.lastModified]));
      if(!uploadDraft||uploadDraft.signature!==signature){
        record=await api('/api/datasets',{name,tags});uploadDraft={id:record.id,signature,uploaded:new Set()};
      }
      for(let i=0;i<files.length;i++){
        const file=files[i]; if(uploadDraft.uploaded.has(file.name))continue;
        $('dataset-progress').textContent=`Uploading ${i+1}/${files.length}: ${file.name}`;
        const response=await fetch(`/api/datasets/${uploadDraft.id}/files/${encodeURIComponent(file.name)}`,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});
        const result=await response.json();if(!response.ok)throw new Error(result.error||'Upload failed');
        uploadDraft.uploaded.add(file.name);
      }
      $('dataset-progress').textContent='Checking arrays and fields…';
      record=await api(`/api/datasets/${uploadDraft.id}/finalize`,spec);uploadDraft=null;
    }else{
      $('dataset-progress').textContent='Checking data…';
      record=await api('/api/datasets',{name,spec,tags});
    }
    $('dataset-progress').textContent=`Added ${record.summary.samples} structures`;
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
  $('dataset-split-progress').textContent=`Validation passed: ${summary.samples} structures`;
}));
$('dataset-split-form').addEventListener('submit',event=>{
  event.preventDefault();datasetOperation(async()=>{
    $('dataset-split-progress').textContent='Creating separate subsets; large datasets may take some time…';
    const result=await api(`/api/datasets/${selectedDataset.id}/split`,{name:text('dataset-split-name'),
      ratios:['train','validation','test'].map(k=>number('dataset-'+k+'-ratio')),
      seed:number('dataset-split-seed'),method:text('dataset-split-method')});
    $('dataset-split-progress').textContent='Created: '+result.datasets.map(d=>`${d.split_role} ${d.summary.samples}`).join(' / ');
  });
});
$('managed-train').addEventListener('change',()=>{if(text('managed-train'))useDataset(text('managed-train'));else setManagedFields();});
for(const split of ['validation','test']) $('managed-'+split).addEventListener('change',()=>{
  if(!text('managed-'+split)){$(split+'-directory').value='';$(split+'-shards').value='';}
  setManagedFields();
});
$('task-type').addEventListener('change',setManagedFields);
init();
