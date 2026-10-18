import json,subprocess


def test_kernel_guard_blocks_new_ip_but_keeps_existing_loopback_and_unix():
    # Apply irreversible seccomp only inside this isolated child, never pytest.
    program='''
import socket,threading,json
from scripts.network_guard import deny_new_ip_sockets,probe_blocked
listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen()
client=socket.create_connection(listener.getsockname());server,_=listener.accept()
deny_new_ip_sockets()
client.sendall(b'proof');assert server.recv(5)==b'proof'
left,right=socket.socketpair();left.sendall(b'x');assert right.recv(1)==b'x'
results=probe_blocked();assert all(results.values())
threads=[]
t=threading.Thread(target=lambda:threads.append(all(probe_blocked().values())));t.start();t.join();assert threads==[True]
print(json.dumps(results))
'''
    result=subprocess.run(['uv','run','--offline','python','-c',program],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert all(json.loads(result.stdout).values())
