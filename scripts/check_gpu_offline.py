"""Run actual primary video processing under kernel network denial.

The model and all its Python descendants arm seccomp after warmup. A separate
analysis worker retains one audited loopback connection, with health keepalive;
new IPv4/IPv6/packet sockets are denied in every processing thread.
"""
import argparse,http.client,json,os,signal,threading,time
from pathlib import Path
from scripts.benchmark_auditor import runtime_signature
from scripts.download_auditor import PROFILES
from scripts.network_guard import deny_new_ip_sockets,existing_ip_sockets,probe_blocked
from service.pipeline.analyze import analyze_detailed


def set_connection_timeout(connection,seconds):
    connection.timeout=seconds
    if connection.sock is not None:connection.sock.settimeout(seconds)


def model_tree(profile):
    directory=PROFILES[profile]['directory'].encode();parents={};commands={}
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            status=(p/'status').read_text();parents[int(p.name)]=int(next(r.split()[1] for r in status.splitlines() if r.startswith('PPid:')))
            commands[int(p.name)]=(p/'cmdline').read_bytes()
        except (OSError,StopIteration):continue
    roots=[pid for pid,c in commands.items() if b'/opt/vllm' in c and b'\0serve\0' in c and directory in c]
    if len(roots)!=1:raise ValueError('Expected one pinned model server')
    selected=set(roots)
    while True:
        more={p for p,parent in parents.items() if parent in selected and p not in selected}
        if not more:break
        selected|=more
    for pid in selected:
        registration=Path(os.environ['OFFLINE_GUARD_LOG_DIR'])/'registered'/(str(pid)+'.json')
        if not registration.is_file():raise ValueError('Model guard registration missing; restart with validation control')
        record=json.loads(registration.read_text());stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if record!=dict(pid=pid,starttime_ticks=int(stat[19]),handler='restrict_network'):raise ValueError('Model guard registration does not match the live process')
    return selected


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--video',type=Path,required=True);p.add_argument('--json',type=Path,required=True);args=p.parse_args(argv)
    started=time.monotonic();result=dict(passed=False,status='failed',model_runtime_signature=runtime_signature(args.profile))
    connection=None;thread=None;stop=threading.Event();lock=threading.Lock();requests_count=[0];ping_errors=[]
    try:
        targets=model_tree(args.profile);proofs=Path(os.environ['OFFLINE_GUARD_LOG_DIR']);armed_epoch=time.time()
        for pid in targets:os.kill(pid,signal.SIGUSR2)
        until=time.monotonic()+20
        while time.monotonic()<until:
            if all((proofs/(str(pid)+'.json')).exists() and json.loads((proofs/(str(pid)+'.json')).read_text())['utc_epoch']>=armed_epoch for pid in targets):break
            time.sleep(.1)
        model_proofs=[json.loads((proofs/(str(pid)+'.json')).read_text()) for pid in sorted(targets)]
        result['model_guards']=model_proofs
        if not all(r['passed'] and r['utc_epoch']>=armed_epoch for r in model_proofs):raise RuntimeError('A model process is not kernel guarded')
        connection=http.client.HTTPConnection('127.0.0.1',8002,timeout=30);connection.connect()
        def ping():
            set_connection_timeout(connection,3)
            connection.request('GET','/health');r=connection.getresponse();r.read(4096)
            if r.status!=200:raise RuntimeError('Model health check failed')
        ping();peers=existing_ip_sockets()
        if not all(p['safe_existing_peer'] for p in peers):raise RuntimeError('Unexpected pre-existing worker network peer')
        def keepalive():
            while not stop.wait(1):
                try:
                    with lock:ping()
                except Exception as exc:ping_errors.append(type(exc).__name__);return
        thread=threading.Thread(target=keepalive,daemon=True);thread.start()
        deny_new_ip_sockets();blocked=probe_blocked();result.update(worker_existing_sockets=peers,worker_blocked=blocked)
        if not all(blocked.values()):raise RuntimeError('Worker network denial failed')
        def request(url,payload,timeout):
            if url!='http://127.0.0.1:8002/v1/chat/completions':raise ValueError('Unexpected endpoint')
            with lock:
                set_connection_timeout(connection,timeout)
                connection.request('POST','/v1/chat/completions',body=payload,headers={'Content-Type':'application/json'})
                response=connection.getresponse();body=response.read(128*1024+1)
                if response.status!=200 or len(body)>128*1024:raise ValueError('Invalid model HTTP response')
                requests_count[0]+=1;return json.loads(body)
        _,analysis=analyze_detailed(args.video,deadline=time.monotonic()+185,requester=request)
        result.update(status='done',engine=analysis['engine'],primary_model_requests=requests_count[0],worker_pid=os.getpid(),ping_errors=ping_errors,
                      passed=analysis['engine']=='mission-v1' and requests_count[0]>0 and not ping_errors)
    except Exception as exc:result['error_code']=type(exc).__name__
    finally:
        stop.set()
        if thread is not None:thread.join(timeout=4)
        if ping_errors or (thread is not None and thread.is_alive()):
            result.update(passed=False,error_code='KeepaliveFailure',ping_errors=list(ping_errors))
        if connection is not None:connection.close()
        result['elapsed_sec']=time.monotonic()-started;args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
