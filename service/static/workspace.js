'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const element = (tag, text, parent, className) => {
    const el = document.createElement(tag);
    if (text !== null) el.textContent = text;
    if (className) el.className = className;
    if (parent) parent.append(el);
    return el;
  };
  const seconds = value => `${String(Math.floor(value / 60)).padStart(2,'0')}:${(value % 60).toFixed(1).padStart(4,'0')}`;
  async function responseJson(response) {
    let data;
    try { data = await response.json(); } catch { throw new Error('The service returned an unreadable response.'); }
    if (!response.ok) {
      const errors = {401:'Access denied. Contact the team for access.',404:'This result is unavailable. Check the job reference or upload a new recording.',413:'Video exceeds the upload limit.',415:'Choose a MOV or MP4 recording.',422:'The video could not be accepted.',503:'The service is busy or unavailable. Please retry shortly.'};
      throw new Error(errors[response.status] || `Request failed (${response.status}). Please retry.`);
    }
    return data;
  }
  function rememberJob(id, started, file = null) {
    try { sessionStorage.setItem('forte-current-job',JSON.stringify({id,started,file})); } catch { /* The result URL works even when storage is unavailable. */ }
    updateResultLinks(id);
  }
  function currentJob() {
    try { return JSON.parse(sessionStorage.getItem('forte-current-job') || 'null'); } catch { return null; }
  }
  function updateResultLinks(id) {
    if(document.body.dataset.offline==='true')return;
    if (typeof id === 'string' && /^[a-f0-9]{12}$/.test(id)) document.querySelectorAll('a[data-current-review],a[data-last-result]').forEach(link => { link.href = `/review?job=${id}`; if(link.hasAttribute('data-last-result'))link.hidden=false; });
  }
  function reviewState(state) {
    document.body.dataset.reviewState=state;
    const labels={empty:'Ready for a recording',selected:'Ready to analyze',uploading:'Uploading',processing:'Analysis in progress',done:'Review ready',failed:'Needs attention'};
    if($('state-label'))$('state-label').textContent=labels[state]||state;
    if($('workbench'))$('workbench').hidden=state==='empty';
    if($('intake-guide'))$('intake-guide').hidden=state!=='empty';
    window.dispatchEvent(new CustomEvent('cashlint:state',{detail:state}));
  }
  function status(title,detail,active=false){
    $('job-status').dataset.quiet='false';$('job-status').hidden=false;$('job-status-title').textContent=title;$('job-status-detail').textContent=detail;
    $('activity-mark').classList.toggle('busy',active);
  }
  window.ForteUI = {$,element,seconds,responseJson,rememberJob,currentJob,reviewState,status};
  const job = currentJob();
  updateResultLinks(job?.id);
})();
