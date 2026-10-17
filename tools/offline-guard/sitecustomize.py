"""Opt-in model-server validation control, inherited by spawned GPU workers."""
import os

if os.environ.get('OFFLINE_GUARD_CONTROL')=='1':
    import json,signal,time
    from pathlib import Path
    from scripts.network_guard import deny_new_ip_sockets,probe_blocked,existing_ip_sockets
    def restrict_network(signum,frame):
        directory=Path(os.environ['OFFLINE_GUARD_LOG_DIR']);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        result=dict(pid=os.getpid(),utc_epoch=time.time(),passed=False)
        try:
            peers=existing_ip_sockets();result['existing_sockets']=peers
            if not all(p['safe_existing_peer'] for p in peers):raise RuntimeError('Pre-existing non-loopback/unconnected network descriptor')
            deny_new_ip_sockets();result['blocked']=probe_blocked();result['passed']=all(result['blocked'].values())
        except Exception as exc:result['error']=type(exc).__name__
        target=directory/(str(os.getpid())+'.json');temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.chmod(0o600);temp.replace(target)
    signal.signal(signal.SIGUSR2,restrict_network)
