"""Linux x86_64 kernel test guard: deny creation of IP/packet sockets.

For isolated validation workers only. Pre-established loopback HTTP and Unix
IPC remain usable. This is irreversible for the process and its threads.
No Python monkeypatch or additional package is used.
"""
import ctypes,errno,ipaddress,platform,socket,struct
from pathlib import Path


class Filter(ctypes.Structure):
    _fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint32)]


class Program(ctypes.Structure):
    _fields_=[('length',ctypes.c_ushort),('filters',ctypes.POINTER(Filter))]


def deny_new_ip_sockets():
    if platform.system()!='Linux' or platform.machine() not in ['x86_64','amd64']:raise RuntimeError('Unverified kernel/architecture for offline guard')
    # seccomp_data offsets: nr0,arch4,args16. Socket=41 on x86_64.
    # Reject other architectures and x32 syscalls rather than bypassing rules.
    instructions=[
        (0x20,0,0,4),(0x15,1,0,0xc000003e),(0x06,0,0,0x80000000),
        (0x20,0,0,0),(0x45,0,1,0x40000000),(0x06,0,0,0x80000000),
    ]
    # Prevent alternate asynchronous socket creation or importing another
    # process's network descriptor. Ordinary file IO/Unix IPC remain available.
    for number in [425,426,427,438]:instructions.extend([(0x15,0,1,number),(0x06,0,0,0x00050000|errno.EPERM)])
    instructions += [
        (0x15,0,4,41),(0x20,0,0,16),(0x15,3,0,socket.AF_INET),
        (0x15,2,0,socket.AF_INET6),(0x15,1,0,socket.AF_PACKET),
        (0x06,0,0,0x7fff0000),(0x06,0,0,0x00050000|errno.EPERM),
    ]
    filters=(Filter*len(instructions))(*(Filter(*row) for row in instructions));program=Program(len(filters),filters)
    libc=ctypes.CDLL(None,use_errno=True);libc.prctl.restype=ctypes.c_int;libc.syscall.restype=ctypes.c_long
    if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'Cannot set no_new_privs')
    # seccomp(SET_MODE_FILTER,TSYNC,&program) synchronizes every existing thread.
    result=libc.syscall(317,1,1,ctypes.byref(program))
    if result!=0:raise OSError(ctypes.get_errno(),'Kernel filter/thread synchronization failed')


def probe_blocked():
    results={}
    for family in [socket.AF_INET,socket.AF_INET6,socket.AF_PACKET]:
        try:
            channel=socket.socket(family,socket.SOCK_STREAM);channel.close();results[str(family)]=False
        except OSError as exc:results[str(family)]=exc.errno==errno.EPERM
    return results


def existing_ip_sockets():
    """Audit this process's pre-existing peers before restricting new sockets."""
    owned=set()
    for fd in Path('/proc/self/fd').iterdir():
        try:link=fd.readlink().as_posix()
        except OSError:continue
        if link.startswith('socket:['):owned.add(link[8:-1])
    def address(value,ipv6):
        raw=bytes.fromhex(value)
        raw=b''.join(raw[i:i+4][::-1] for i in range(0,len(raw),4))
        return ipaddress.ip_address(raw)
    def local(value):return value.is_loopback or bool(getattr(value,'ipv4_mapped',None) and value.ipv4_mapped.is_loopback)
    result=[]
    for protocol in ['tcp','tcp6','udp','udp6','raw','raw6']:
        path=Path('/proc/self/net')/protocol
        if not path.exists():continue
        for line in path.read_text().splitlines()[1:]:
            fields=line.split()
            if len(fields)<10 or fields[9] not in owned:continue
            own=address(fields[1].split(':')[0],protocol.endswith('6'));peer=address(fields[2].split(':')[0],protocol.endswith('6'));state=fields[3]
            safe=(protocol.startswith('tcp') and state=='0A') or (local(peer) and (local(own) or own.is_unspecified))
            result.append(dict(protocol=protocol,state=state,local=str(own),peer=str(peer),safe_existing_peer=safe))
    return result
