/* Offline HTML contract: open real file URLs, select synthetic video, use both pages.
 * node scripts/check_frontend_export.cjs <export-dir> <synthetic-mp4> [playwright-module] [browser-exe]
 */
'use strict';
const path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const {chromium}=require(process.argv[4]||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.argv[5]?{executablePath:process.argv[5]}:{})});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],external=[];let checks=0;
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))external.push(r.url())});
 const check=(value,message)=>{assert.ok(value,message);checks++;};
 const file=name=>pathToFileURL(path.join(process.argv[2],name)).href;
 await page.goto(file('fortebank-review.html'));await page.evaluate(()=>document.fonts.ready);
 check(await page.locator('body').getAttribute('data-offline')==='true','export is explicitly offline');
 const originalURL=page.url();await page.locator('#file').setInputFiles(process.argv[3]);await page.waitForFunction(()=>document.querySelector('#video').readyState>=1);
 check(await page.locator('#video').evaluate(el=>el.duration>0&&Number.isFinite(el.duration)),'export plays local synthetic MP4');
 await page.locator('#submit').click();check(await page.locator('#job-status-title').innerText()==='Design preview only','offline analysis clearly labelled');
 check(page.url()===originalURL,'local file selection does not replace the URL with an application route');
 check(await page.locator('.event').count()===0,'no invented offline findings');
 await page.locator('.theme-toggle').click();check(await page.locator('html').getAttribute('data-theme')==='light','offline theme works');
 await page.locator('.sidebar a[href="evaluation.html"]').click();await page.waitForURL('**/evaluation.html');
 await page.waitForFunction(()=>document.querySelector('#accuracy-value').textContent==='66.7%');check(true,'embedded metrics render without fetch');
 await page.selectOption('#recipe','0');check(await page.locator('#accuracy-value').innerText()==='40.0%','offline comparison works');
 check((await page.locator('#measured-headline').innerText()).includes('12 of 30'),'offline measured headline updates');
 await page.goto(file('results.html'));check(await page.locator('#upload').count()===1,'legacy exported alias contains unified Review');
 check(await page.locator('#workbench').isHidden(),'exported alias does not load a stored real job');
 check(errors.length===0,'no script errors');check(external.length===0,'no external HTTP requests');
 console.log(JSON.stringify({checks,errors,externalRequests:external.length},null,2));await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
