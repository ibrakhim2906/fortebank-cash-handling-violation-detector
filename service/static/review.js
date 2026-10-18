'use strict';
(() => {
  const {$,responseJson,rememberJob,currentJob,seconds,reviewState,status}=window.ForteUI;
  const results=window.CashLintResults,offline=document.body.dataset.offline==='true';
  let file=null,objectUrl=null,busy=false,job=null,jobStarted=null,metadata=null,localOnly=false;
  const video=$('video'),zone=$('drop-zone');
  function replaceUrl(path){if(!offline)history.replaceState(null,'',path);}
  function setBusy(value){
    busy=value;$('file').disabled=busy;$('source-file').disabled=busy&&!job;$('replace-recording').disabled=busy;$('retry-result').disabled=busy;
    $('submit').disabled=busy||!file||localOnly;
    $('submit-label').textContent=busy?'Uploading…':document.body.dataset.reviewState==='failed'&&!job?'Try upload again':'Analyze recording';
    $('submit').hidden=Boolean(job);$('replace-recording').hidden=!file&&!job;
    $('replace-recording').textContent=job?'New recording':'Change video';
  }
  function releaseVideo(){if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=null;video.pause();video.removeAttribute('src');video.load();results.setRecording(false);}
  function validFile(selected){
    if(!/\.(mov|mp4)$/i.test(selected.name)){status('Unsupported recording','Choose a MOV or MP4 video.');return false;}
    if(!selected.size||selected.size>200*1024**2){status('Recording cannot be accepted','Choose a non-empty video up to 200 MiB.');return false;}
    return true;
  }
  function attach(selected,{restore=false}={}){
    releaseVideo();file=selected;localOnly=restore;
    objectUrl=URL.createObjectURL(file);video.src=objectUrl;$('playback-error').hidden=true;
    $('video-file-name').textContent=file.name;$('video-clock').textContent='00:00';
    $('media-empty-title').textContent='Loading local playback…';$('media-empty-copy').textContent='Your selected recording is opening in this browser.';
    $('attach-recording').hidden=true;$('playback-note').hidden=!restore;
    if(restore)$('playback-note').textContent='User-selected local playback. Original API evidence remains the source for this result.';
    results.setRecording(true);video.load();
    $('file-duration').textContent=`${(file.size/1024**2).toFixed(1)} MiB · Reading video metadata…`;
  }
  function setFile(selected){
    if(busy||!selected)return;
    if(!validFile(selected))return;
    results.clear();job=null;jobStarted=null;metadata=null;localOnly=false;
    replaceUrl('/review');$('upload').classList.add('has-recording');$('file-label').textContent='SELECTED RECORDING';
    $('file-detail').textContent=selected.name;attach(selected);reviewState('selected');setBusy(false);
  }
  function restoreSource(selected){
    if((busy&&!job)||!selected||!validFile(selected))return;
    if(metadata&&(selected.name!==metadata.name||selected.size!==metadata.size)){
      status('Choose the original recording',`This job was submitted as ${metadata.name}. The selected filename or size differs; existing findings were preserved.`);return;
    }
    attach(selected,{restore:true});$('file-detail').textContent=selected.name;setBusy(busy);
    if(!metadata)$('playback-note').textContent='User-selected local playback; original-file identity is unverified. Use the API evidence for this result.';
    status('Local playback attached','Findings and original evidence are preserved. File metadata does not establish source identity.');
  }
  async function loadJob(id,started,source=null){
    job=id;jobStarted=started;metadata=source;setBusy(true);$('upload').classList.add('has-recording');
    if(/^[a-f0-9]{12}$/.test(id)){
      rememberJob(id,started,source);replaceUrl(`/review?job=${id}`);
      if(!file){$('file-label').textContent='RECORDED ANALYSIS';$('file-detail').textContent=source?.name||'Recording submitted to the analysis service';$('file-duration').textContent=source?`${(source.size/1024**2).toFixed(1)} MiB · Select the original for local playback`:'Original evidence is available with the completed result';}
    }
    if(!file){$('media-empty-title').textContent='Add the original for playback.';$('media-empty-copy').textContent='Result links restore the evidence. Select the original file to also play the recording locally.';$('attach-recording').hidden=false;}
    try{await results.load(id,started);}finally{setBusy(false);}
  }
  function uploadVideo(selected,started){
    return new Promise((resolve,reject)=>{
      const request=new XMLHttpRequest();request.open('POST','/ui/analyze');request.timeout=Math.max(1,300000-(Date.now()-started));
      request.upload.addEventListener('progress',event=>{
        if(event.lengthComputable){$('upload-progress').hidden=false;$('upload-progress').value=event.loaded/event.total*100;
          $('job-status-detail').textContent=`${(event.loaded/1024**2).toFixed(1)} of ${(event.total/1024**2).toFixed(1)} MiB transferred. Waiting for service acceptance.`;}
      });
      request.addEventListener('load',async()=>{try{resolve(await responseJson({ok:request.status>=200&&request.status<300,status:request.status,json:async()=>JSON.parse(request.responseText)}));}catch(error){reject(error);}});
      request.addEventListener('error',()=>reject(new Error('The upload connection was interrupted. Check the connection before trying again.')));
      request.addEventListener('timeout',()=>reject(new Error('The upload deadline expired. No completed analysis can be inferred.')));
      const body=new FormData();body.append('file',selected);request.send(body);
    });
  }
  $('file').addEventListener('change',()=>setFile($('file').files[0]));
  $('source-file').addEventListener('change',()=>restoreSource($('source-file').files[0]));
  $('replace-recording').addEventListener('click',()=>{if(!busy){$('file').value='';$('file').click();}});
  $('new-recording').addEventListener('click',()=>{if(!busy){$('file').value='';$('file').click();}});
  $('attach-recording').addEventListener('click',()=>{$('source-file').value='';$('source-file').click();});
  video.addEventListener('error',()=>{
    if(!file||!video.getAttribute('src'))return;
    results.setRecording(false);$('playback-error').hidden=false;$('video-clock').textContent='Playback unavailable';
    $('media-empty-title').textContent='Local playback is unavailable.';$('media-empty-copy').textContent=job?'Use the original evidence frames to inspect this result.':'You can still analyze this recording and inspect the returned original evidence.';
    $('file-duration').textContent=`${(file.size/1024**2).toFixed(1)} MiB · Browser playback unavailable`;
  });
  video.addEventListener('loadedmetadata',()=>{
    results.setRecording(true);$('playback-error').hidden=true;$('file-duration').textContent=`${(file.size/1024**2).toFixed(1)} MiB · ${Number.isFinite(video.duration)?seconds(video.duration)+' duration':'Duration unavailable'}`;
  });
  video.addEventListener('timeupdate',()=>{$('video-clock').textContent=seconds(video.currentTime);});
  ['dragenter','dragover'].forEach(name=>zone.addEventListener(name,event=>{event.preventDefault();if(!busy)zone.classList.add('dragover');}));
  ['dragleave','drop'].forEach(name=>zone.addEventListener(name,event=>{event.preventDefault();zone.classList.remove('dragover');}));
  zone.addEventListener('drop',event=>{if(busy)return;if(event.dataTransfer.files.length!==1){status('Choose one recording','Analyze one video at a time.');return;}setFile(event.dataTransfer.files[0]);});
  $('upload').addEventListener('submit',async event=>{
    event.preventDefault();if(busy||!file||job||localOnly)return;
    if(offline){status('Design preview only','Run the CashLint service to analyze recordings. This HTML supports local video preview and contains no sample findings.');return;}
    results.clear();setBusy(true);reviewState('uploading');results.empty('Uploading the recording.','Analysis begins after the service accepts the upload. Your selected video stays on this page.');
    const started=Date.now(),timer=setInterval(()=>{$('elapsed').textContent=`${Math.floor((Date.now()-started)/1000)}s elapsed`;},1000);
    try{
      status('Uploading recording…','Transferring the selected recording to the analysis service.',true);
      const accepted=await uploadVideo(file,started);
      if(!/^[a-f0-9]{12}$/.test(accepted.job_id))throw new Error('The service did not return a valid job identifier.');
      clearInterval(timer);$('upload-progress').hidden=true;
      await loadJob(accepted.job_id,started,{name:file.name,size:file.size});
    }catch(error){reviewState('failed');status('Upload not confirmed',error.message);results.empty('The upload was not confirmed.','Check the connection and try again. No completed analysis can be inferred.');}
    finally{clearInterval(timer);$('upload-progress').hidden=true;setBusy(false);$('activity-mark').classList.remove('busy');}
  });
  $('retry-result').addEventListener('click',()=>{if(!busy&&job)loadJob(job,jobStarted,metadata);});
  window.addEventListener('cashlint:state',()=>setBusy(busy));
  document.querySelectorAll('a[data-current-review].active').forEach(link=>link.addEventListener('click',event=>event.preventDefault()));
  window.addEventListener('pagehide',()=>{if(objectUrl)URL.revokeObjectURL(objectUrl);});
  window.addEventListener('pageshow',event=>{if(event.persisted&&file){objectUrl=URL.createObjectURL(file);video.src=objectUrl;video.load();}});
  reviewState('empty');setBusy(false);
  const stored=currentJob(),supplied=new URLSearchParams(location.search).get('job');
  const initial=supplied??(location.pathname==='/results'?stored?.id:null);
  if(initial&&!offline)loadJob(initial,stored?.id===initial&&Number.isFinite(stored.started)?stored.started:Date.now(),stored?.id===initial?stored.file:null);
  else if(location.pathname==='/results')replaceUrl('/review');
})();
