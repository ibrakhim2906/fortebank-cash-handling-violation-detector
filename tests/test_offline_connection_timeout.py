import http.client,http.server,threading,time
from scripts.check_gpu_offline import set_connection_timeout


def test_proof_reuses_connected_socket_with_the_actual_model_call_timeout():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(.1);self.send_response(200);self.send_header('Content-Length','0');self.end_headers()
        def log_message(self,*args):pass
    server=http.server.HTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.handle_request);worker.start()
    connection=http.client.HTTPConnection(*server.server_address,timeout=.01)
    try:
        connection.connect();set_connection_timeout(connection,1)
        connection.request('GET','/');response=connection.getresponse();response.read()
        assert response.status==200
    finally:connection.close();worker.join(timeout=2);server.server_close()
