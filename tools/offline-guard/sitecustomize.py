"""Opt-in model-server validation control, inherited by spawned GPU workers."""
import os

if os.environ.get('OFFLINE_GUARD_CONTROL')=='1':
    import json,signal,time
    from pathlib import Path
    from scripts.network_guard import deny_new_ip_sockets,probe_blocked,existing_ip_sockets,local_interface_addresses
    def restrict_network(signum,frame):
        directory=Path(os.environ['OFFLINE_GUARD_LOG_DIR']);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        result=dict(pid=os.getpid(),utc_epoch=time.time(),passed=False)
        try:
            result['local_interface_addresses']=sorted(str(v) for v in local_interface_addresses())
            peers=existing_ip_sockets();result['existing_sockets']=peers
            if not all(p['safe_existing_peer'] for p in peers):raise RuntimeError('Pre-existing non-local/unconnected network descriptor')
            deny_new_ip_sockets();result['blocked']=probe_blocked();result['passed']=all(result['blocked'].values())
        except Exception as exc:result['error']=type(exc).__name__
        target=directory/(str(os.getpid())+'.json');temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.chmod(0o600);temp.replace(target)
    signal.signal(signal.SIGUSR2,restrict_network)
    def register_control():
        # EngineCore uses setproctitle, which can overwrite the environment
        # memory exposed by /proc/PID/environ. Record actual handler setup,
        # including forked workers, instead of treating that buffer as truth.
        directory=Path(os.environ['OFFLINE_GUARD_LOG_DIR'])/'registered';directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        record=dict(pid=os.getpid(),starttime_ticks=int(stat[19]),handler=signal.getsignal(signal.SIGUSR2).__name__)
        target=directory/(str(os.getpid())+'.json');temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(record)+'\n');temporary.chmod(0o600);temporary.replace(target)
    register_control()
    os.register_at_fork(after_in_child=register_control)
