import json,os,subprocess,sys
from pathlib import Path
import pytest


@pytest.mark.skipif(sys.platform!='linux',reason='Linux kernel validation control')
def test_forked_model_worker_has_live_registration_and_signal_guard(tmp_path):
    root=Path(__file__).resolve().parents[1]
    code='''
import json,os,signal,socket,time
from pathlib import Path
root=Path(os.environ['OFFLINE_GUARD_LOG_DIR'])
pid=os.fork()
if pid==0:
    time.sleep(15)
    os._exit(0)
try:
    marker=root/'registered'/(str(pid)+'.json')
    until=time.monotonic()+8
    while not marker.exists() and time.monotonic()<until:time.sleep(.02)
    registration=json.loads(marker.read_text())
    stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    assert registration==dict(pid=pid,starttime_ticks=int(stat[19]),handler='restrict_network')
    os.kill(pid,signal.SIGUSR2)
    target=root/(str(pid)+'.json')
    until=time.monotonic()+8
    while not target.exists() and time.monotonic()<until:time.sleep(.02)
    proof=json.loads(target.read_text())
    assert proof['passed'] and all(proof['blocked'].values())
    channel=socket.socket(socket.AF_INET,socket.SOCK_STREAM);channel.close()
    print(json.dumps(dict(passed=True)))
finally:
    os.kill(pid,signal.SIGTERM);os.waitpid(pid,0)
'''
    env=dict(os.environ,PYTHONPATH=str(root/'tools/offline-guard')+':'+str(root),OFFLINE_GUARD_CONTROL='1',OFFLINE_GUARD_LOG_DIR=str(tmp_path))
    done=subprocess.run(['uv','run','--offline','--no-sync','python','-c',code],cwd=root,env=env,capture_output=True,text=True,timeout=20)
    assert done.returncode==0,done.stderr
    assert json.loads(done.stdout)['passed']
