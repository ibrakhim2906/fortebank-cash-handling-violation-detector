'use strict';
(() => {
  const $=id=>document.getElementById(id),pct=value=>(value*100).toFixed(1)+'%';
  const element=(tag,text,parent,className)=>{const el=document.createElement(tag);if(text!==null)el.textContent=text;if(className)el.className=className;if(parent)parent.append(el);return el;};
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  let snapshot,hasPlayed=false,frame=null,finishCount=null,barAnimations=[];
  function finishMotion(){
    if(frame!==null)cancelAnimationFrame(frame);frame=null;
    if(finishCount){const finish=finishCount;finishCount=null;finish();}
    barAnimations.forEach(animation=>animation.cancel());barAnimations=[];
    $('kpis').setAttribute('aria-busy','false');
  }
  function animateInitial(r,checkpoint){
    if(hasPlayed)return;hasPlayed=true;
    if(motion.matches){$('kpis').setAttribute('aria-busy','false');return;}
    const values=[['accuracy-value',r.accuracy*100,'%'],['recall-value',r.recall*100,'%'],['timing-value',r.hit_2s*100,'%'],['runtime-value',checkpoint.worst_sec,'s']];
    $('kpis').setAttribute('aria-busy','true');
    finishCount=()=>{values.forEach(([id,value,unit])=>{$(id).textContent=value.toFixed(1)+unit;});};
    values.forEach(([id,value,unit])=>{$(id).textContent='0.0'+unit;});
    const started=performance.now();
    const tick=now=>{
      const progress=Math.min(1,(now-started)/500),eased=1-(1-progress)**3;
      values.forEach(([id,value,unit])=>{$(id).textContent=(value*eased).toFixed(1)+unit;});
      if(progress<1)frame=requestAnimationFrame(tick);else{frame=null;finishMotion();}
    };
    frame=requestAnimationFrame(tick);
    document.querySelectorAll('.localization-fill').forEach(fill=>{barAnimations.push(fill.animate([{transform:'scaleX(0)'},{transform:'scaleX(1)'}],{duration:500,easing:'cubic-bezier(0.215,0.61,0.355,1)'}));});
  }
  motion.addEventListener('change',()=>{if(motion.matches)finishMotion();});
  const luminance=rgb=>rgb.map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;}).reduce((sum,v,i)=>sum+v*[.2126,.7152,.0722][i],0);
  function inkFor(tone,surface,intensity){
    const mixed=tone.map((v,i)=>v*intensity+surface[i]*(1-intensity)),l=luminance(mixed);
    const dark=luminance([17,20,24]);
    if((l+.05)/(dark+.05)>=4.5)return '#111418';
    if(1.05/(l+.05)>=4.5)return '#ffffff';
    return '#000000';
  }
  function matrix(r){
    const parent=$('outcomes');parent.replaceChildren();
    const table=element('table',null,parent,'matrix-table');
    element('caption','Rows: human reference · Columns: detector',table);
    const head=element('tr',null,element('thead',null,table));
    const corner=element('th',null,head,'matrix-axis');element('span','Human reference',corner,'sr-only');
    for(const name of ['Possible violation','No violation']){const th=element('th',name,head);th.scope='col';}
    const body=element('tbody',null,table),max=Math.max(1,r.tp,r.fn,r.fp,r.tn);
    const rows=[['Violation',[r.tp,'Detected','tp'],[r.fn,'Missed / unprocessed','fn']],['No violation',[r.fp,'False alarms','fp'],[r.tn,'Correctly clear','tn']]];
    for(const [name,...cells] of rows){
      const tr=element('tr',null,body),th=element('th',name,tr);th.scope='row';
      for(const [count,label,kind] of cells){
        const cell=element('td',null,tr,`matrix-cell ${kind}`);
        const intensity=count===0?0:.16+.84*count/max;
        cell.style.setProperty('--intensity',`${(intensity*100).toFixed(1)}%`);
        const lightTone=kind==='tp'?[163,21,81]:[113,128,147],darkTone=kind==='tp'?[210,122,163]:[135,148,162];
        // Compute ink for both themes so theme changes preserve contrast without rerendering.
        cell.style.setProperty('--cell-ink-light',inkFor(lightTone,[255,255,255],intensity));
        cell.style.setProperty('--cell-ink-dark',inkFor(darkTone,[25,29,34],intensity));
        element('strong',String(count),cell);element('span',label,cell);
      }
    }
    const outside=r.n_videos-r.tp-r.fn-r.fp-r.tn;
    $('matrix-warning').hidden=!(r.failed||outside);
    $('matrix-warning').textContent=outside?`${outside} unprocessed · ${r.n_videos} videos total`:`${r.failed} failed predictions · See methodology.`;
    $('outcome-foot').textContent=`${r.fp} false alarms · ${r.failed} failed/missing primary predictions remain in the denominator.${r.failed>0?' Positive recordings without a completed prediction are included under missed / unprocessed; they are not clear verdicts.':''}${outside>0?` ${outside} unprocessed ${outside===1?'negative recording is':'negative recordings are'} outside the four classified cells.`:''}`;
  }
  function configurationComparison(){
    const target=$('configuration-comparison');target.replaceChildren();
    const selected=Number($('recipe').value);
    snapshot.recipes.forEach((r,i)=>{
      const item=element('div',null,target,'configuration-item');
      item.dataset.selected=String(i===selected);
      element('dt',r.short_name,item);
      element('dd',`${r.tp}/${r.tp+r.fn}`,item);
    });
    target.hidden=false;
  }
  function renderEvaluation(){
    finishMotion();
    const index=Number($('recipe').value),r=snapshot.recipes[index],base=snapshot.recipes[0],last=snapshot.recipes.at(-1);
    $('overview-title').textContent=`Detection rose from ${base.tp}/${base.tp+base.fn} to ${last.tp}/${last.tp+last.fn} across ${snapshot.recipes.length} iterations.`;
    $('measured-headline').textContent=`${r.correct} of ${r.n_videos} verdicts correct.`;
    $('scope-total').textContent=`${r.n_videos} previously reviewed videos`;
    $('evaluation-priority-title').textContent=`${r.fn} missed recordings. ${r.fp} false alarms.`;
    $('evaluation-next').textContent=`${r.fn} positive recordings were missed and ${r.fp} false alarms were raised. ${r.localized_2s} of ${r.labelled_events} annotated events were localized within two seconds. Use alerts to prioritize human review.`;
    for(const [prefix,field] of [['accuracy','accuracy'],['recall','recall'],['timing','hit_2s']]){
      $(`${prefix}-value`).textContent=pct(r[field]);
      const delta=(r[field]-base[field])*100;
      $(`${prefix}-delta`).textContent=index?`${delta>=0?'+':''}${delta.toFixed(1)} pp`:'Reference';
      $(`${prefix}-delta`).setAttribute('aria-label',index?`${delta>=0?'+':''}${delta.toFixed(1)} percentage points versus original 8B`:'Original 8B reference');
    }
    $('accuracy-detail').textContent=`${r.correct} / ${r.n_videos} correct verdicts`;
    $('recall-detail').textContent=`${r.tp} / ${r.tp+r.fn} positive videos`;
    $('timing-detail').textContent=`${r.localized_2s} / ${r.labelled_events} annotated events`;
    $('target-gap').textContent=`${pct(r.accuracy)} measured · ${((snapshot.target_accuracy-r.accuracy)*100).toFixed(1)} pp to target`;
    $('outcome-title').textContent=`${r.tp} of ${r.tp+r.fn} positive videos were detected.`;matrix(r);
    $('localization-title').textContent=`${r.labelled_events-r.localized_2s} of ${r.labelled_events} events still lack a precise timestamp.`;
    $('localization-takeaway').textContent=`${r.localized_2s} of ${r.labelled_events} within ±2s${r.localized_2s>base.localized_2s?`, up from ${base.localized_2s}`:''}.`;
    $('localization-chart').replaceChildren();
    snapshot.recipes.forEach((item,i)=>{
      const row=element('div',null,$('localization-chart'),`localization-row${i===index?' selected':''}`),name=element('div',item.short_name,row,'localization-name');

      const track=element('div',null,row,'localization-track'),fill=element('div',null,track,'localization-fill');fill.style.width=`${item.hit_2s*100}%`;
      track.setAttribute('role','img');track.setAttribute('aria-label',`${item.localized_2s} of ${item.labelled_events} events within two seconds`);
      element('div',`${item.localized_2s}/${item.labelled_events}`,row,'localization-value');
    });
    const checkpoint=snapshot.checkpoint;
    $('runtime-value').textContent=`${checkpoint.worst_sec.toFixed(1)}s`;
    $('runtime-delta').textContent=`${(checkpoint.limit_sec-checkpoint.worst_sec).toFixed(1)}s below the 300s limit`;
    $('runtime-detail').textContent=`${checkpoint.clean_uploads}/${checkpoint.clean_uploads} recordings via public API · ${checkpoint.api_jobs} jobs · ${checkpoint.failed_jobs} failed`;
    configurationComparison();animateInitial(r,checkpoint);
  }
  async function loadEvaluation(){
    try{
      const embedded=$('export-evaluation-data');
      if(embedded)snapshot=JSON.parse(embedded.textContent);
      else{const response=await fetch('/static/evaluation.json',{cache:'no-store',signal:AbortSignal.timeout(15000)});if(!response.ok)throw new Error('Metrics unavailable');snapshot=await response.json();}
      if(!Array.isArray(snapshot.recipes)||snapshot.recipes.length<3||!snapshot.recipes.every(r=>Number.isFinite(r.accuracy)&&r.n_videos>0))throw new Error('Invalid metric snapshot');
      $('recipe').replaceChildren();snapshot.recipes.forEach((r,i)=>{const option=element('option',r.name,$('recipe'));option.value=i;});$('recipe').value=String(snapshot.recipes.length-1);$('recipe').disabled=false;
      $('snapshot').textContent=snapshot.snapshot_label;
      $('source-caption').textContent=`Snapshot: ${snapshot.snapshot_label}. Sources: ${snapshot.recipes.map(r=>r.source).join('; ')}. SHA-256 source hashes are included in the bundled aggregate snapshot.`;
      renderEvaluation();
    }catch{
      finishMotion();$('metrics-error').hidden=false;$('overview-title').textContent='Measured development performance.';$('measured-headline').textContent='The recorded evaluation is currently unavailable.';$('snapshot').textContent='Snapshot unavailable';$('target-gap').textContent='Target only · measured data unavailable';
    }finally{$('metrics-loading').hidden=true;}
  }
  $('recipe').addEventListener('change',renderEvaluation);
  loadEvaluation();
})();
