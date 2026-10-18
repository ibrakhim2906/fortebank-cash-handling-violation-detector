'use strict';
(() => {
  const {$,element,seconds,responseJson,reviewState,status}=window.ForteUI;
  let events=[],active=-1,frameIndex=0,loading=false,recording=false,mode='recording';
  const offline=document.body.dataset.offline==='true';
  function viewer(next){
    if(next==='evidence'&&!events.length)return;
    mode=next;
    $('recording-tab').setAttribute('aria-selected',String(mode==='recording'));
    $('evidence-tab').setAttribute('aria-selected',String(mode==='evidence'));
    $('video-shell').hidden=mode!=='recording'||!recording;
    $('media-empty').hidden=mode!=='recording'||recording;
    $('evidence-inspector').hidden=mode!=='evidence';
    $('viewer-source').textContent=mode==='evidence'?'ORIGINAL FRAME':'LOCAL PLAYBACK';
    if(mode==='evidence')$('video').pause();
    if(events[active]){
      const frame=events[active].evidence_frames[frameIndex]||events[active].evidence_frames[0];
      $('selected-caption').textContent=mode==='evidence'?`Original evidence · ${seconds(frame.timestamp_sec)} from video start. Select the image to enlarge.`:'Event timecodes refer to seconds from the recording’s start.';
    }
    $('recording-tab').tabIndex=mode==='recording'?0:-1;$('evidence-tab').tabIndex=mode==='evidence'?0:-1;
  }
  function setRecording(value){recording=value;viewer(recording?'recording':events.length?'evidence':'recording');}
  function empty(title,copy){$('findings-empty').hidden=false;$('findings-empty-title').textContent=title;$('findings-empty-copy').textContent=copy;}
  function clear(){
    events=[];active=-1;frameIndex=0;
    if($('frame-dialog').open)$('frame-dialog').close();
    $('events').replaceChildren();$('verdict').hidden=true;$('result-summary').hidden=true;$('analysis-note').hidden=true;
    $('event-count').textContent='—';$('event-position').textContent='—';$('evidence-total').textContent='';
    $('prev-event').disabled=true;$('next-event').disabled=true;$('evidence-tab').disabled=true;
    $('event-timeline').hidden=true;$('event-timeline').replaceChildren();$('timeline-header').hidden=true;
    $('selected-time').textContent='—';$('selected-caption').hidden=true;$('selected-frame').removeAttribute('src');
    $('frame-picker').replaceChildren();$('recommendation').hidden=true;$('job-reference').hidden=true;
    $('job-status').hidden=true;$('retry-result').hidden=true;$('elapsed').textContent='';
    $('upload-progress').hidden=true;$('frame-error').hidden=true;
    empty('Findings appear here.','Analyze the recording to see reported events.');
    viewer('recording');
  }
  function selectFrame(index){
    const event=events[active];if(!event)return;
    frameIndex=index;const frame=event.evidence_frames[index];
    $('frame-error').hidden=true;$('inspect-frame').hidden=false;
    $('selected-frame').src=`data:image/jpeg;base64,${frame.image_base64}`;
    $('selected-frame').alt=`Original evidence at ${seconds(frame.timestamp_sec)}: ${event.type}`;
    $('selected-time').textContent=seconds(frame.timestamp_sec);
    $('frame-position').textContent=`Frame ${index+1} of ${event.evidence_frames.length}`;
    $('prev-frame').disabled=index===0;$('next-frame').disabled=index===event.evidence_frames.length-1;
    $('selected-caption').textContent=mode==='evidence'?`Original evidence · ${seconds(frame.timestamp_sec)} from video start. Select the image to enlarge.`:'Event timecodes refer to seconds from the recording’s start.';
    $('selected-caption').hidden=false;
    $('frame-picker').querySelectorAll('button').forEach((button,i)=>button.setAttribute('aria-pressed',String(i===index)));
  }
  function activate(index,{seek=false,focus=false,evidence=false}={}){
    if(index<0||index>=events.length)return;
    active=index;const event=events[index];
    document.querySelectorAll('.event').forEach((card,i)=>{
      card.classList.toggle('selected-event',i===index);
      card.querySelector('.event-select').setAttribute('aria-expanded',String(i===index));
      card.querySelector('.event-detail').hidden=i!==index;
    });
    $('event-position').textContent=`${index+1} / ${events.length}`;
    $('prev-event').disabled=index===0;$('next-event').disabled=index===events.length-1;
    $('event-timeline').querySelectorAll('button').forEach((button,i)=>button.setAttribute('aria-pressed',String(i===index)));
    $('frame-picker').replaceChildren();
    event.evidence_frames.forEach((frame,i)=>{
      const button=element('button',`${i+1} · ${seconds(frame.timestamp_sec)}`,$('frame-picker'),'frame-choice');
      button.type='button';button.setAttribute('aria-label',`Evidence frame ${i+1}, ${seconds(frame.timestamp_sec)}`);
      button.addEventListener('click',()=>selectFrame(i));
    });
    $('evidence-total').textContent=String(event.evidence_frames.length);
    if(evidence||!recording)viewer('evidence');
    selectFrame(0);
    const video=$('video');
    if(seek&&recording&&video.readyState>=1&&Number.isFinite(video.duration))video.currentTime=Math.max(0,Math.min(event.timecode_sec,video.duration));
    if(focus)document.querySelectorAll('.event-select')[index].focus({preventScroll:true});
  }
  function sourceLinks(text,parent){
    const urls=[...new Set(text.match(/https?:\/\/[^\s<>"'\u00ab\u00bb)]+/g)||[])];
    if(!urls.length){element('p','No source link supplied. Verify the applicable procedure manually.',parent);return;}
    urls.forEach(raw=>{try{
      const url=new URL(raw.replace(/[.,;]+$/,''));if(!['https:','http:'].includes(url.protocol))return;
      const p=element('p',null,parent),link=element('a',`Open source · ${url.hostname} ↗`,p);
      link.href=url.href;link.target='_blank';link.rel='noopener noreferrer';
    }catch{/* Unparseable text is not promoted to a link. */}});
  }
  function validResult(data){
    return ['violation','no_violation'].includes(data.verdict)&&Array.isArray(data.violations)
      &&(data.verdict!=='no_violation'||data.violations.length===0)
      &&(data.verdict!=='violation'||data.violations.length>0)
      &&data.violations.every(e=>typeof e.type==='string'&&typeof e.description==='string'
        &&Number.isFinite(e.timecode_sec)&&e.timecode_sec>=0&&Array.isArray(e.evidence_frames)
        &&e.evidence_frames.length>=1&&e.evidence_frames.length<=3
        &&e.evidence_frames.every(f=>Number.isFinite(f.timestamp_sec)&&f.timestamp_sec>=0
          &&typeof f.image_base64==='string'&&/^[A-Za-z0-9+/=\r\n]+$/.test(f.image_base64)));
  }
  function recommendation(label,title,copy){$('recommendation').hidden=false;$('recommendation-label').textContent=label;$('recommendation-title').textContent=title;$('recommendation-copy').textContent=copy;}
  function showResult(data){
    if(!validResult(data))throw new Error('The service returned an incomplete result. No verdict can be confirmed.');
    events=data.violations;const flagged=data.verdict==='violation';
    const degraded=typeof data.analysis?.engine==='string'&&data.analysis.engine.includes('degraded');
    $('findings-empty').hidden=true;$('result-summary').hidden=false;$('verdict').hidden=false;
    $('verdict').className=`verdict${flagged?'':' clean'}`;
    $('verdict').textContent=flagged?'Signs of possible violations detected':'No violation detected within the model’s capabilities';
    $('event-count').textContent=String(events.length);
    if(degraded){$('analysis-note').hidden=false;$('analysis-note').textContent='Limited fallback analysis was used. Manually review the recording; this result does not establish reliable action recognition.';}
    if(!flagged)empty('No events reported.','The detector did not report a possible violation. This does not prove that all required actions were completed or visible.');
    $('event-timeline').hidden=!flagged;$('timeline-header').hidden=!flagged;$('evidence-tab').disabled=!flagged;
    events.forEach((event,index)=>{
      const card=element('article',null,$('events'),'event');
      const button=element('button',null,card,'event-select');button.type='button';button.id=`event-select-${index}`;button.setAttribute('aria-controls',`event-detail-${index}`);button.setAttribute('aria-expanded','false');
      element('span',seconds(event.timecode_sec),button,'event-time');element('span',event.type,button,'event-title');element('span','⌄',button,'event-chevron');
      const detail=element('div',null,card,'event-detail');detail.id=`event-detail-${index}`;detail.setAttribute('role','region');detail.setAttribute('aria-labelledby',button.id);detail.hidden=true;
      element('div','OBSERVATION',detail,'detail-label');element('p',event.description,detail);
      const source=element('div',null,detail,'source-note');element('div','PROCEDURE / SOURCE',source,'detail-label');sourceLinks(event.description,source);
      const evidence=element('button',`Inspect original evidence · ${event.evidence_frames.length} ${event.evidence_frames.length===1?'frame':'frames'} ↗`,detail,'event-evidence');evidence.type='button';evidence.addEventListener('click',()=>{activate(index,{evidence:true});$('evidence-tab').focus({preventScroll:true});document.querySelector('.viewer-panel').scrollIntoView({block:'start'});});
      button.addEventListener('click',()=>activate(index,{seek:true}));
      const marker=element('button',seconds(event.timecode_sec),$('event-timeline'),'time-link');marker.type='button';marker.setAttribute('aria-label',`${event.type}, ${seconds(event.timecode_sec)}`);marker.addEventListener('click',()=>activate(index,{seek:true}));
    });
    if(flagged)activate(0);
    status(degraded?'Completed with limited fallback':'Analysis complete',flagged?'Select a finding to inspect its timecode, explanation and source evidence.':'No events reported. Verify critical steps manually.');
    $('job-status').dataset.quiet='true';
    recommendation('NEXT ACTION',degraded?'Manually review this recording.':flagged?'Verify the flagged moments.':'No alerts reported. Review critical steps manually.',degraded?'The primary recognition path was unavailable. Inspect the original recording and applicable procedures.':flagged?`${events.length} possible ${events.length===1?'event requires':'events require'} review. Original evidence supports inspection; procedure applicability needs human confirmation.`:'A negative model result does not establish that every required action was completed.');
    reviewState('done');
  }
  async function load(job,started=Date.now()){
    if(loading||!job)return;
    clear();
    if(!/^[a-f0-9]{12}$/.test(job)){reviewState('failed');status('Invalid result reference','Use a valid result link or choose a recording.');return;}
    loading=true;$('job-reference').hidden=false;$('job-reference').textContent=`Job ${job}`;
    reviewState('processing');empty('Retrieving the analysis.','Findings and original evidence will appear here when a completed result is available.');
    const remaining=Math.max(15000,300000-(Date.now()-started)),controller=new AbortController(),deadline=setTimeout(()=>controller.abort(),remaining);
    let timer=null;
    const clock=()=>{$('elapsed').textContent=`${Math.max(0,Math.floor((Date.now()-started)/1000))}s elapsed`;};
    try{
      status('Loading result…','Retrieving the recorded analysis for this job.',true);
      while(true){
        const data=await responseJson(await fetch(`/ui/jobs/${encodeURIComponent(job)}`,{cache:'no-store',signal:controller.signal}));
        if(data.status==='done'){showResult(data);$('elapsed').textContent='';break;}
        if(data.status==='failed')throw new Error('Analysis did not complete. This does not mean the recording is clear.');
        if(data.status!=='processing')throw new Error('The service returned an unknown processing state.');
        if(Date.now()-started>=300000)throw new DOMException('Timed out','AbortError');
        status('Analyzing recording…','Processing is in progress. Upload and queue time count toward the five-minute limit.',true);
        empty('Analysis is in progress.','The service has not returned findings yet. You can inspect the recording while you wait.');clock();if(!timer)timer=setInterval(clock,1000);
        await new Promise(resolve=>setTimeout(resolve,1500));if(controller.signal.aborted)throw new DOMException('Timed out','AbortError');
      }
    }catch(error){
      reviewState('failed');status('Analysis not confirmed',error.name==='AbortError'?'The five-minute wait expired. Check again to retrieve a completed result.':error.message);
      empty('No completed analysis confirmed.','A failed or timed-out request is not a clean verdict. Check this job again or review the recording manually.');
      recommendation('ANALYSIS NOT CONFIRMED','Check the result again or review manually.','No completed verdict is confirmed for this job. Retry retrieves the same job without uploading the recording again.');
      $('retry-result').hidden=false;
    }finally{clearTimeout(deadline);clearInterval(timer);loading=false;$('activity-mark').classList.remove('busy');}
  }
  $('recording-tab').addEventListener('click',()=>viewer('recording'));
  $('evidence-tab').addEventListener('click',()=>{viewer('evidence');selectFrame(frameIndex);});
  document.querySelector('.viewer-tabs').addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();const buttons=[...document.querySelectorAll('.viewer-tabs button:not([disabled])')];
    const current=buttons.indexOf(document.activeElement),index=event.key==='Home'?0:event.key==='End'?buttons.length-1:(current+(event.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length;
    buttons[index].focus();buttons[index].click();
  });
  $('prev-event').addEventListener('click',()=>activate(active-1,{seek:true,focus:true}));
  $('next-event').addEventListener('click',()=>activate(active+1,{seek:true,focus:true}));
  $('prev-frame').addEventListener('click',()=>selectFrame(frameIndex-1));$('next-frame').addEventListener('click',()=>selectFrame(frameIndex+1));
  $('selected-frame').addEventListener('error',()=>{if(events[active]){$('frame-error').hidden=false;$('inspect-frame').hidden=true;}});
  $('inspect-frame').addEventListener('click',()=>{
    const event=events[active];if(!event)return;const frame=event.evidence_frames[frameIndex];
    $('expanded-frame').src=`data:image/jpeg;base64,${frame.image_base64}`;$('expanded-frame').alt=$('selected-frame').alt;
    $('expanded-caption').textContent=`${event.type} · ${seconds(frame.timestamp_sec)} from video start`;$('frame-dialog').showModal();
  });
  $('close-dialog').addEventListener('click',()=>$('frame-dialog').close());
  window.CashLintResults={clear,load,setRecording,viewer,empty,get hasEvents(){return events.length>0;}};
})();
