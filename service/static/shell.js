/* Restore desktop sidebar state before CSS; mobile opens as a closed drawer. */
'use strict';
(() => {
  const root=document.documentElement;
  const mobile=window.matchMedia('(max-width: 960px)');
  const key='cashlint-sidebar';
  function desktopState(){try{return localStorage.getItem(key)==='closed'?'closed':'open';}catch{return 'open';}}
  root.dataset.sidebar=mobile.matches?'closed':desktopState();
  document.addEventListener('DOMContentLoaded',()=>{
    const sidebar=document.getElementById('workspace-sidebar');
    const toggle=document.getElementById('sidebar-toggle');
    const backdrop=document.querySelector('.sidebar-backdrop');
    const main=document.querySelector('main');
    function setOpen(open,{remember=false,focus=false}={}){
      const returnFocus=!open&&sidebar.contains(document.activeElement);
      root.dataset.sidebar=open?'open':'closed';
      sidebar.hidden=!open;sidebar.inert=!open;
      toggle.setAttribute('aria-expanded',String(open));
      toggle.title=open?'Close sidebar':'Open sidebar';
      backdrop.hidden=!(open&&mobile.matches);
      main.inert=open&&mobile.matches;
      document.body.classList.toggle('sidebar-modal',open&&mobile.matches);
      if(open&&mobile.matches){sidebar.setAttribute('role','dialog');sidebar.setAttribute('aria-modal','true');}
      else{sidebar.removeAttribute('role');sidebar.removeAttribute('aria-modal');}
      if(remember&&!mobile.matches){try{localStorage.setItem(key,open?'open':'closed');}catch{/* Optional preference. */}}
      if(open&&focus&&mobile.matches)sidebar.querySelector('nav a').focus();
      if(returnFocus)toggle.focus();
    }
    toggle.addEventListener('click',()=>setOpen(root.dataset.sidebar!=='open',{remember:true,focus:true}));
    backdrop.addEventListener('click',()=>{setOpen(false);toggle.focus();});
    sidebar.querySelectorAll('nav a').forEach(link=>link.addEventListener('click',()=>{if(mobile.matches)setOpen(false);}));
    document.addEventListener('keydown',event=>{
      // Let an evidence dialog handle Escape before navigation does.
      if(document.querySelector('dialog[open]'))return;
      if(root.dataset.sidebar!=='open')return;
      if(event.key==='Escape'){event.preventDefault();setOpen(false,{remember:true});toggle.focus();return;}
      if(event.key==='Tab'&&mobile.matches){
        const items=[...sidebar.querySelectorAll('a[href],button:not([disabled])')];
        const first=items[0],last=items[items.length-1],active=document.activeElement;
        if(event.shiftKey&&(active===first||!sidebar.contains(active))){event.preventDefault();last.focus();}
        else if(!event.shiftKey&&(active===last||!sidebar.contains(active))){event.preventDefault();first.focus();}
      }
    });
    mobile.addEventListener('change',()=>setOpen(mobile.matches?false:desktopState()==='open'));
    window.addEventListener('pageshow',()=>setOpen(mobile.matches?false:desktopState()==='open'));
    setOpen(root.dataset.sidebar==='open');
  });
})();
