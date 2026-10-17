import ipaddress,socket
import pytest
from scripts.network_guard import existing_ip_sockets,local_interface_addresses,same_machine_address


def test_exact_owned_address_does_not_allow_an_entire_private_network():
    owned={ipaddress.ip_address('172.22.0.2')}
    assert same_machine_address(ipaddress.ip_address('172.22.0.2'),owned)
    assert same_machine_address(ipaddress.ip_address('::ffff:172.22.0.2'),owned)
    assert same_machine_address(ipaddress.ip_address('::1'),owned)
    assert not same_machine_address(ipaddress.ip_address('172.22.0.3'),owned)
    assert not same_machine_address(ipaddress.ip_address('8.8.8.8'),owned)


def test_real_same_namespace_tcp_connection_is_local_even_without_loopback_ip():
    owned=local_interface_addresses();address=next((a for a in owned if a.version==4 and not a.is_loopback),None)
    if address is None:pytest.skip('No non-loopback IPv4 interface in this namespace')
    with socket.socket() as server,socket.socket() as client:
        server.bind((str(address),0));server.listen(1);client.connect(server.getsockname())
        accepted,_=server.accept()
        with accepted:
            matching=[p for p in existing_ip_sockets() if p['state']=='01' and p['local']==p['peer']==str(address)]
            assert len(matching)>=2 and all(p['safe_existing_peer'] for p in matching)
