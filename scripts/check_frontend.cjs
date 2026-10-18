/* Jury-flow browser checks. All media and analysis responses here are synthetic.
 * node scripts/check_frontend.cjs <local-url> <output-dir> [playwright-module] [browser-exe]
 * Place the explicitly synthetic fixture.jpg in output-dir before running.
 */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.DATATHON_PLAYWRIGHT||process.argv[4]||'playwright');
const base=(process.argv[2]||'http://127.0.0.1:8018').replace(/\/$/,''),out=path.resolve(process.argv[3]||'reports/frontend-review');
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,...((process.env.DATATHON_BROWSER_PATH||process.argv[5])?{executablePath:process.env.DATATHON_BROWSER_PATH||process.argv[5]}:{})});
 const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
 let checks=0;const errors=[],contrastSamples=[];page.on('pageerror',e=>errors.push(e.message));
 const check=(value,message)=>{assert.ok(value,message);checks++;};
 const shot=async name=>page.screenshot({path:path.join(out,name+'.png'),fullPage:true});
 const theme=async value=>{
  if(await page.locator('html').getAttribute('data-theme')===value)return;
  if(!await page.locator('.sidebar').isVisible())await page.locator('#sidebar-toggle').click();
  await page.locator('.theme-toggle').click();
  if(page.viewportSize().width<=960&&await page.locator('.sidebar').isVisible())await page.keyboard.press('Escape');
 };
 const fits=async label=>check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),label+' fits viewport');
 async function matrix(label){
  for(const t of ['dark','light']){
   await theme(t);
   for(const width of [1440,1024,768,390,320]){
    await page.setViewportSize({width,height:900});await fits(`${label} ${t} ${width}`);
    if(label==='evaluation'&&width<=390){
     check(await page.locator('#kpis').evaluate(el=>getComputedStyle(el).gridTemplateColumns.split(' ').length===2),'mobile KPIs use two columns');
     check(await page.locator('#kpis').evaluate(el=>el.getBoundingClientRect().height<390),'mobile KPI summary stays compact');
    }
    check(await page.locator('#sidebar-toggle').isVisible(),'sidebar toggle reachable');
    if(width<=960){
     await page.waitForFunction(()=>document.documentElement.dataset.sidebar==='closed');
     check(await page.locator('.sidebar').isHidden(),'mobile drawer closed');await page.locator('#sidebar-toggle').click();
     check(await page.locator('main').evaluate(el=>el.inert),'mobile background inert');
     check(await page.locator('.sidebar-bottom .theme-toggle').isVisible(),'theme switch visible in drawer');
     await page.keyboard.press('Shift+Tab');check(await page.locator('.theme-toggle').evaluate(el=>el===document.activeElement),'drawer traps backward focus');
     await page.keyboard.press('Tab');check(await page.locator('.sidebar nav a').first().evaluate(el=>el===document.activeElement),'drawer wraps forward focus');
     await page.keyboard.press('Escape');check(await page.locator('.sidebar').isHidden(),'Escape closes drawer');
    }
   }
   await page.setViewportSize({width:390,height:844});await shot(label+'-'+t+'-mobile');
   await page.setViewportSize({width:1440,height:1000});await shot(label+'-'+t+'-desktop');
  }
  await theme('dark');
 }
 await page.goto(base+'/review');await page.evaluate(()=>document.fonts.ready);
 check(await page.title()==='Review recording · CashLint','review document');
 check(await page.locator('.sidebar nav a').count()===2,'two product destinations');
 check(await page.locator('#kpis,.breadcrumb,.product-name,.nav-index').count()===0,'no metric/breadcrumb/page-number clutter in review');
 check(await page.locator('#submit').isDisabled(),'no analysis without a file');
 check(await page.locator('#workbench').isHidden(),'empty review does not show fabricated results');
 check(await page.locator('footer').evaluate(el=>Math.abs(el.getBoundingClientRect().bottom-innerHeight)<2),'empty Review footer reaches viewport bottom');
 check(await page.locator('html').getAttribute('data-theme')==='dark','dark default');
 check(await page.locator('.sidebar-bottom .theme-toggle').count()===1,'theme anchored at sidebar bottom');
 check(await page.locator('.theme-label').innerText()==='Switch to light','theme button labels its action');
 check(await page.locator('#intake-guide .stage-label svg').count()===3,'empty review strip has three stage icons');
 check(await page.locator('#saved-review,#open-review,#review-reference').count()===0,'review has no technical job-ID entry panel');
 await page.locator('#sidebar-toggle').click();check(await page.locator('.sidebar').isHidden(),'desktop sidebar closes');
 check(await page.locator('main').evaluate(el=>getComputedStyle(el).marginLeft)==='0px','closed sidebar releases content width');
 await page.reload();check(await page.locator('.sidebar').isHidden(),'sidebar preference survives refresh');await page.locator('#sidebar-toggle').click();
 await page.locator('.theme-toggle').focus();await page.keyboard.press('Space');check(await page.locator('html').getAttribute('data-theme')==='light','keyboard theme toggle');
 await page.goto(base+'/evaluation');check(await page.locator('html').getAttribute('data-theme')==='light','theme persists across documents');
 await page.waitForFunction(()=>document.querySelector('#accuracy-value').textContent==='66.7%');
 check((await page.locator('#measured-headline').innerText()).includes('20 of 30'),'measured verdicts remain a supporting stat');
 check(await page.locator('#overview-title').innerText()==='Detection rose from 2/16 to 12/16 across three iterations.','requested hero uses existing detection counts');
 check(JSON.stringify(await page.locator('.matrix-cell strong').allTextContents())==='["12","4","6","8"]','confusion matrix preserves default counts and orientation');
 check(await page.locator('.matrix-table caption').innerText()==='Rows: human reference · Columns: detector','matrix axes are explicit');
 check(await page.locator('.comparison-title').innerText()==='Positive videos detected','progression states the measured unit');
 check(await page.locator('.configuration-item').nth(1).evaluate(el=>getComputedStyle(el,'::before').content.includes('→')),'comparison uses connecting arrows');
 check(await page.locator('#localization-takeaway').innerText()==='5 of 16 within ±2s, up from 1.','localization takeaway matches the snapshot');
 check(await page.locator('.localization-panel').evaluate(el=>{const axis=el.querySelector('.chart-scale');return el.getBoundingClientRect().bottom-axis.getBoundingClientRect().bottom<40}),'localization card ends shortly after the bars and axis');

 check(await page.locator('.sparkline,.stacked-bar,#step-chart').count()===0,'oversized plot and old decorative charts removed');
 check(JSON.stringify(await page.locator('#configuration-comparison dd').allTextContents())==='["2/16","6/16","12/16"]','compact comparison preserves all detection counts');
 check(JSON.stringify(await page.locator('#configuration-comparison dt').allTextContents())==='["Original 8B","Source 8B","32B proposal"]','configuration values retain their labels');
 check(await page.locator('#configuration-comparison').evaluate(el=>el.getBoundingClientRect().height<80),'comparison uses a compact row');


 await page.locator('.methodology summary').click();
 check(await page.locator('#target-gap').innerText()==='66.7% measured · 23.3 pp to target','target is distinct from result');
 check((await page.locator('#runtime-delta').innerText()).includes('223.5s'),'runtime headroom is retained in methodology');
 await page.locator('.methodology summary').click();
 await page.selectOption('#recipe','0');check(await page.locator('#accuracy-value').innerText()==='40.0%','comparison updates live');
 check((await page.locator('#measured-headline').innerText()).includes('12 of 30'),'supporting stat follows selected configuration');
 check(JSON.stringify(await page.locator('.matrix-cell strong').allTextContents())==='["2","14","3","10"]','original configuration matrix preserves counts');
 check(await page.locator('#matrix-warning').isVisible(),'missing-result accounting remains visible');
 check(await page.locator('#matrix-warning').innerText()==='1 unprocessed · 30 videos total','original 8B has its separate unprocessed note');
 check(await page.locator('.matrix-cell strong').evaluateAll(els=>els.reduce((total,el)=>total+Number(el.textContent),0)+1)===30,'matrix cells plus the separate unprocessed recording account for all 30');

 await page.locator('.methodology summary').click();
 check((await page.locator('#outcome-foot').innerText()).includes('1 unprocessed negative recording'),'unprocessed negative stays outside classified matrix');
 await page.locator('.methodology summary').click();

 await page.selectOption('#recipe','1');check(await page.locator('#recall-value').innerText()==='37.5%','second configuration');
 await page.selectOption('#recipe','2');
 const contrast=async()=>page.evaluate(()=>{
  const ctx=document.createElement('canvas').getContext('2d');
  const rgb=value=>{ctx.clearRect(0,0,1,1);ctx.fillStyle=value;ctx.fillRect(0,0,1,1);return [...ctx.getImageData(0,0,1,1).data].slice(0,3);};
  const lum=value=>rgb(value).map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4}).reduce((s,v,i)=>s+v*[.2126,.7152,.0722][i],0);
  const ratio=(a,b)=>{a=lum(a);b=lum(b);return(Math.max(a,b)+.05)/(Math.min(a,b)+.05)};
  const css=getComputedStyle(document.documentElement),accent=css.getPropertyValue('--accent').trim();
  return {accent,accentSurface:ratio(accent,css.getPropertyValue('--surface').trim()),accentCanvas:ratio(accent,css.getPropertyValue('--canvas').trim()),cells:[...document.querySelectorAll('.matrix-cell')].map(el=>{const s=getComputedStyle(el);return ratio(s.color,s.backgroundColor)})};
 });
 for(const t of ['dark','light']){await theme(t);for(const option of ['0','1','2']){await page.selectOption('#recipe',option);const c=await contrast();contrastSamples.push({theme:t,configuration:option,...c});check(Math.min(...c.cells)>=4.5,'matrix labels have 4.5:1 contrast in '+t+' config '+option);if(t==='dark')check(c.accentSurface>=3&&c.accentCanvas>=3,'dark accent exceeds 3:1');else check(c.accent==='#a31551','light theme retains Forte magenta');}}
 await theme('dark');
 await page.goto(base+'/evaluation');await page.waitForFunction(()=>document.querySelector('#recipe').disabled===false);
 const growing=await page.evaluate(()=>document.getAnimations().filter(a=>a.effect.target.classList.contains('localization-fill')).map(a=>a.effect.getTiming().duration));
 check(growing.length===3&&growing.every(ms=>ms>=400&&ms<=600),'bars animate once on load for 400–600ms');
 await page.selectOption('#recipe','0');await page.waitForTimeout(650);
 check(await page.locator('#accuracy-value').innerText()==='40.0%','configuration change cancels old count-up');
 await page.selectOption('#recipe','2');check(await page.locator('#accuracy-value').innerText()==='66.7%','count-up does not replay on selection');check(await page.evaluate(()=>document.getAnimations().length===0),'bar growth does not replay on selection');
 const reduced=await browser.newPage({reducedMotion:'reduce'});await reduced.goto(base+'/evaluation');await reduced.waitForFunction(()=>document.querySelector('#recipe').disabled===false);
 check(await reduced.locator('#accuracy-value').innerText()==='66.7%','reduced motion shows final numbers immediately');
 check(await reduced.evaluate(()=>document.getAnimations().length===0),'reduced motion disables bar animations');await reduced.close();
 await matrix('evaluation');
 await page.goto(base+'/review');await matrix('empty');
 await page.setViewportSize({width:390,height:844});await page.locator('#sidebar-toggle').click();await page.locator('.sidebar .nav-item.active').click();
 check(await page.locator('.sidebar').isHidden(),'active navigation closes drawer without reload');
 await page.locator('#sidebar-toggle').click();await page.locator('.sidebar-backdrop').click({position:{x:300,y:100}});check(await page.locator('.sidebar').isHidden(),'outside click closes drawer');
 await page.setViewportSize({width:1440,height:1000});
 // Record a genuine, playable synthetic portrait MP4 in the already installed browser.
 const media=await page.evaluate(async()=>{
  const canvas=document.createElement('canvas');canvas.width=464;canvas.height=848;const ctx=canvas.getContext('2d');
  const draw=()=>{ctx.fillStyle='#dfe3e7';ctx.fillRect(0,0,464,848);ctx.fillStyle='#27313b';ctx.font='bold 24px sans-serif';ctx.fillText('SYNTHETIC UI FIXTURE',30,60);ctx.font='20px sans-serif';ctx.fillText('No bank footage',30,95);ctx.strokeStyle='#a31551';ctx.lineWidth=4;ctx.strokeRect(72,240,320,300);ctx.fillStyle='#a31551';ctx.fillRect(90+(Date.now()%1500)/8,360,40,40);};draw();
  const stream=canvas.captureStream(10),mime='video/mp4';if(!MediaRecorder.isTypeSupported(mime))throw Error('Existing browser lacks synthetic MP4 recording support');
  const recorder=new MediaRecorder(stream,{mimeType:mime}),chunks=[];recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
  const completed=new Promise(resolve=>recorder.onstop=resolve);recorder.start(100);const timer=setInterval(draw,80);await new Promise(r=>setTimeout(r,3300));recorder.stop();await completed;clearInterval(timer);stream.getTracks().forEach(t=>t.stop());
  const bytes=new Uint8Array(await new Blob(chunks,{type:mime}).arrayBuffer());let binary='';bytes.forEach(x=>binary+=String.fromCharCode(x));return {base64:btoa(binary),mime};
 });
 const videoBytes=Buffer.from(media.base64,'base64');fs.writeFileSync(path.join(out,'synthetic.mp4'),videoBytes);
 const choose=async(name='synthetic.mp4',buffer=videoBytes)=>page.locator('#file').setInputFiles({name,mimeType:media.mime,buffer});
 await choose('invalid.txt');check(await page.locator('#job-status-title').innerText()==='Unsupported recording','invalid extension rejected');
 check(await page.locator('#submit').isDisabled(),'invalid file cannot submit');
 let phase='positive',polls=0,uploads=0;
 const jpeg=fs.readFileSync(path.join(out,'fixture.jpg')).toString('base64');
 const event=(type,time,frames=1,description='Synthetic observation only. https://adilet.zan.kz/eng This is a UI fixture, not a bank finding.')=>({type,timecode_sec:time,description,evidence_frames:Array.from({length:frames},(_,i)=>({timestamp_sec:time+i*.1,image_base64:jpeg}))});
 const events=[event('Synthetic fixture <img src=x onerror=alert(1)>',.5,3),event('Synthetic second review point',1.2),event('Synthetic long bilingual finding · Проверка наблюдаемого действия и применимости процедуры',2,2,'Synthetic description without a public source link. '.repeat(5))];
 await page.route('**/ui/analyze',async route=>{uploads++;polls=0;await route.fulfill({json:{job_id:'abc012345678'}});});
 await page.route('**/ui/jobs/*',async route=>{
  if(++polls===1){await route.fulfill({json:{status:'processing',verdict:null,violations:[]}});return;}
  if(phase==='failed'){await route.fulfill({json:{status:'failed',verdict:null,violations:[]}});return;}
  if(phase==='malformed'){await route.fulfill({json:{status:'done',verdict:'no_violation',violations:[{}]}});return;}
  if(phase==='negative'){await route.fulfill({json:{status:'done',verdict:'no_violation',violations:[]}});return;}
  if(phase==='broken-frame'){await route.fulfill({json:{status:'done',verdict:'violation',violations:[{...events[0],evidence_frames:[{timestamp_sec:.5,image_base64:'AAAA'}]}]}});return;}
  await route.fulfill({json:{status:'done',verdict:'violation',analysis:{engine:'mission-v1-degraded'},violations:events}});
 });
 await choose();await page.waitForFunction(()=>document.querySelector('#video').readyState>=1);
 check(await page.locator('#video').evaluate(el=>Number.isFinite(el.duration)&&el.duration>2.5),'synthetic MP4 decodes with finite duration');
 await page.setViewportSize({width:1366,height:768});check(await page.locator('#submit').evaluate(el=>el.getBoundingClientRect().bottom<innerHeight),'analyze action visible on a short laptop');await shot('selected-laptop');
 await page.setViewportSize({width:1440,height:1000});
 await page.evaluate(()=>{window.sameReviewDocument=true;window.originalBlob=document.querySelector('#video').src;});
 await page.locator('#submit').click();await page.waitForURL('**/review?job=abc012345678');
 check(await page.evaluate(()=>window.sameReviewDocument===true),'accepted upload remains in document');
 check(await page.evaluate(()=>document.querySelector('#video').src===window.originalBlob),'local video survives upload');
 await page.waitForFunction(()=>document.querySelector('#job-status-title').textContent==='Analyzing recording…');
 check(await page.locator('#verdict').isHidden(),'processing cannot look clean');check(await page.locator('#file').isDisabled(),'file locked during processing');
 await shot('processing-dark-desktop');
 await page.waitForFunction(()=>document.querySelector('#job-status-title').textContent==='Completed with limited fallback');
 check(uploads===1,'one upload request');check(await page.locator('#submit').isHidden(),'completed job cannot be accidentally resubmitted');
 check(await page.locator('.event').count()===3,'multiple events render');check(await page.locator('.event-title img').count()===0,'model wording cannot inject markup');
 check(await page.locator('.event-detail:visible').count()===1,'one expanded event');check(await page.locator('#analysis-note').isVisible(),'fallback clearly qualified');
 await page.locator('.event-select').nth(1).click();
 check(await page.locator('#event-position').innerText()==='2 / 3','event selection synchronized');
 await page.waitForFunction(()=>Math.abs(document.querySelector('#video').currentTime-1.2)<.1);check(true,'event selection seeks the actual playable recording');
 await page.locator('#next-event').click();check(await page.locator('#event-position').innerText()==='3 / 3','next finding');
 check(await page.locator('#next-event').isDisabled(),'last finding has no next action');
 check((await page.locator('.event-detail:visible').innerText()).includes('No source link supplied'),'missing source is explicit');
 await page.locator('#prev-event').click();await page.locator('#prev-event').click();
 await page.locator('#evidence-tab').click();
 check(await page.locator('#evidence-inspector').isVisible(),'evidence tab uses main viewer');
 check(await page.locator('#video-shell').isHidden(),'one primary media pane');
 check(await page.locator('#selected-frame').getAttribute('src')==='data:image/jpeg;base64,'+jpeg,'original response evidence used');
 await page.locator('#next-frame').click();check(await page.locator('#frame-position').innerText()==='Frame 2 of 3','next original frame');
 await page.locator('#prev-frame').click();await page.locator('#inspect-frame').click();check(await page.locator('#frame-dialog').isVisible(),'evidence enlarges');
 await page.keyboard.press('Escape');check(await page.locator('#frame-dialog').isHidden(),'Escape dismisses dialog');
 check(await page.locator('#inspect-frame').evaluate(el=>el===document.activeElement),'modal returns focus');
 await matrix('completed');
 await page.locator('#recording-tab').click();await page.locator('#recording-tab').focus();await page.keyboard.press('ArrowRight');check(await page.locator('#evidence-tab').getAttribute('aria-selected')==='true','keyboard switches media tabs');
 await page.locator('#recording-tab').click();check(!(await page.locator('#selected-caption').innerText()).includes('Original evidence'),'recording caption follows active media');await shot('completed-recording-desktop');
 await page.setViewportSize({width:1366,height:768});await page.evaluate(()=>scrollTo({top:0,behavior:'instant'}));
 check(await page.locator('#video').evaluate(el=>el.getBoundingClientRect().bottom<innerHeight),'completed recording controls visible on short laptop');await shot('completed-laptop');
 await page.setViewportSize({width:1440,height:1000});
 await choose('next.mp4');check(await page.locator('.event').count()===0,'new file clears old findings');check(await page.locator('#recommendation').isHidden(),'new file clears recommendation');check(!new URL(page.url()).searchParams.has('job'),'new file clears job URL');
 await page.goto(base+'/review?job=abc012345678');await page.waitForFunction(()=>document.querySelector('#event-count').textContent==='3');
 check(await page.locator('#evidence-inspector').isVisible(),'result link restores evidence without a local file');
 await page.locator('#recording-tab').click();check(await page.locator('#attach-recording').isVisible(),'restored result offers original-file reselection');
 await page.locator('#source-file').setInputFiles({name:'different.mp4',mimeType:media.mime,buffer:videoBytes});
 check(await page.locator('#job-status-title').innerText()==='Choose the original recording','mismatched stored metadata rejected');
 check(await page.locator('.event').count()===3,'mismatch preserves findings');
 await page.locator('#source-file').setInputFiles({name:'synthetic.mp4',mimeType:media.mime,buffer:videoBytes});await page.waitForFunction(()=>document.querySelector('#video').readyState>=1);
 check(await page.locator('#job-status-title').innerText()==='Local playback attached','original playback reattached');
 check(await page.locator('.event').count()===3,'reattachment preserves findings');check(page.url().includes('job=abc012345678'),'reattachment preserves job');
 await page.locator('.event-select').nth(1).click();await page.waitForFunction(()=>Math.abs(document.querySelector('#video').currentTime-1.2)<.1);check(true,'reattached recording seeks to selected event');
 await page.goto(base+'/evaluation');await page.locator('.sidebar a[data-current-review]').click();await page.waitForURL('**/review?job=abc012345678');await page.waitForFunction(()=>document.querySelector('#event-count').textContent==='3');check(true,'navigation returns to latest job');
 await page.goto(base+'/results?job=abc012345678');await page.waitForURL('**/review?job=abc012345678');check(await page.locator('#upload').count()===1,'legacy alias opens unified review');
 phase='negative';polls=0;await page.goto(base+'/review?job=abc012345678');await page.waitForFunction(()=>document.querySelector('#job-status-title').textContent==='Analysis complete');
 check((await page.locator('#verdict').innerText()).startsWith('No violation detected'),'negative result qualified');check(await page.locator('.event').count()===0,'no stale positive findings');check(await page.locator('#evidence-tab').isDisabled(),'negative has no fabricated evidence');await shot('negative-desktop');
 phase='failed';polls=0;await page.reload();await page.waitForFunction(()=>document.querySelector('#job-status-title').textContent==='Analysis not confirmed');
 check(await page.locator('#verdict').isHidden(),'failed job has no verdict');check(await page.locator('#retry-result').isVisible(),'retry offered');await shot('failure-desktop');
 phase='malformed';await page.locator('#retry-result').click();await page.waitForFunction(()=>document.querySelector('#job-status-title').textContent==='Analysis not confirmed');check(await page.locator('#verdict').isHidden(),'malformed result rejected');check(uploads===1,'retry never reuploads');
 phase='broken-frame';polls=0;await page.reload();await page.locator('#frame-error').waitFor({state:'visible'});check(await page.locator('#frame-error').evaluate(el=>el.getBoundingClientRect().height>=220),'broken frame preserves viewer stage');check(true,'broken original image gets a visible fallback');check(await page.locator('#inspect-frame').isHidden(),'broken image cannot enlarge');
 await page.goto(base+'/results?job=invalid');check(await page.locator('#job-status-title').innerText()==='Invalid result reference','invalid deep link handled');
 await page.goto(base+'/review');await choose('unplayable.mov',Buffer.from('Synthetic invalid-video fixture'));await page.locator('#playback-error').waitFor({state:'visible'});check(await page.locator('#submit').isEnabled(),'unplayable local video can still be analyzed');
 // Simulate an old processing job without waiting five minutes or resubmitting it.
 await page.evaluate(()=>sessionStorage.setItem('forte-current-job',JSON.stringify({id:'abc012345678',started:Date.now()-310000})));
 polls=0;await page.goto(base+'/review?job=abc012345678');await page.waitForFunction(()=>document.querySelector('#job-status-title').textContent==='Analysis not confirmed');check((await page.locator('#job-status-detail').innerText()).includes('five-minute'),'expired original budget handled');
 await page.route('**/static/evaluation.json',route=>route.fulfill({status:503,body:'unavailable'}));await page.goto(base+'/evaluation');await page.locator('#metrics-error').waitFor({state:'visible'});check(await page.locator('#accuracy-value').innerText()==='—','missing metrics never fabricated');
 await page.emulateMedia({reducedMotion:'reduce'});check(await page.evaluate(()=>getComputedStyle(document.documentElement).scrollBehavior)==='auto','reduced-motion scrolling respected');
 check(errors.length===0,'no JavaScript errors: '+JSON.stringify(errors));
 const report={checks,errors,contrastSamples,scope:'Two-page jury flow; playable synthetic portrait MP4, synthetic multi-event responses, no real inference.'};
 fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
