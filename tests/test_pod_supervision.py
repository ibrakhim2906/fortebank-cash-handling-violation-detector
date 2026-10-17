"""Real supervisor replacement with a harmless sleeping child, no GPU/server."""
import os,re,shutil,subprocess,time
from pathlib import Path
import pytest


@pytest.mark.skipif(not Path('/proc').exists() or not shutil.which('flock'),reason='Linux/flock lifecycle test')
def test_child_does_not_keep_supervisor_lock_after_owner_exits(tmp_path):
    text=Path('scripts/pod_setup.sh').read_text().replace('cd /workspace/datathon-kit','cd "'+str(tmp_path)+'"')
    text=re.sub(r'(?ms)^service\)\n.*?^  ;;','service)\n  exec sleep 60\n  ;;',text,count=1)
    script=tmp_path/'setup.sh';script.write_text(text)
    directory=tmp_path/'data/processed/mission-runtime';directory.mkdir(parents=True)
    processes=[]
    try:
        for iteration in range(2):
            process=subprocess.Popen(['bash',str(script),'supervise','service'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);processes.append(process)
            stop=time.monotonic()+5;child=None
            while time.monotonic()<stop:
                pidfile=directory/'service-supervisor.pid';childfile=directory/'service-child.pid'
                if pidfile.exists() and pidfile.read_text().strip()==str(process.pid) and childfile.exists():
                    candidate=int(childfile.read_text())
                    if Path(f'/proc/{candidate}/cmdline').exists() and b'sleep' in Path(f'/proc/{candidate}/cmdline').read_bytes():child=candidate;break
                time.sleep(.02)
            assert child is not None and process.poll() is None
            assert Path(f'/proc/{process.pid}/fd/9').exists()
            assert not Path(f'/proc/{child}/fd/9').exists()
            process.terminate();process.wait(timeout=3)
    finally:
        for process in processes:
            if process.poll() is None:process.terminate();process.wait(timeout=3)
        childfile=directory/'service-child.pid'
        if childfile.exists():
            try:os.kill(int(childfile.read_text()),15)
            except ProcessLookupError:pass
