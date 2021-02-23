"""

A HTTP-server allowing (limited) access to the Deep Learning Toolbox
via a web browser (Experimental!).


"""

# standard imports
import http.server
import urllib

# toolbox imports
from dltb.datasource import Datasource
from util.error import handle_exception

# logging
import logging
logger = logging.getLogger(__name__)


class ToolboxHandler(http.server.BaseHTTPRequestHandler):

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_GET(self):
        """Respond to a GET request."""
        parsed_path = urllib.parse.urlparse(self.path)
        message_parts = [
                'CLIENT VALUES:',
                'client_address=%s (%s)' % (self.client_address,
                                            self.address_string()),
                'command=%s' % self.command,
                'path=%s' % self.path,
                'real path=%s' % parsed_path.path,
                'query=%s' % parsed_path.query,
                'request_version=%s' % self.request_version,
                '',
                'SERVER VALUES:',
                'server_version=%s' % self.server_version,
                'sys_version=%s' % self.sys_version,
                'protocol_version=%s' % self.protocol_version,
                '',
                'HEADERS RECEIVED:',
                ]
        for name, value in sorted(self.headers.items()):
            message_parts.append('%s=%s' % (name, value.rstrip()))
        message_parts.append('')
        message = "<html>"
        message += "  <head>"
        message += "    <title>Deep Learning Toolbox</title>"
        message += "  </head>"
        message += "  <body>"
        message += "    <h1>Datasources</h1>"
        message += "    <ul>"
        for key in Datasource.instance_register.keys():
            message += f'      <li><a href="datasource/{key}">{key}</a></li>'
        message += "    </ul>"
        message += "    <pre>"
        message += '\r\n'.join(message_parts)
        message += "    </pre>"
        message += "  </body>"
        message += "</html>"

        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(message.encode())
        except BrokenPipeError as exception:
            handle_exception(exception)

    def log_message(self, *args, **kwargs):
        print(f"HTTP Server log_message: {args}, {kwargs}")

    def log_request(self, *args, **kwargs):
        logger.info(f"log_request: {args}, {kwargs}")

    def log_error(self, *args, **kwargs):
        logger.error("log_error: {args}, {kwargs}")

class ToolboxServer(http.server.HTTPServer):
    pass

class Server:

    def __init__(self, server_class=ToolboxServer,
                 handler_class=ToolboxHandler) -> None:
        # Servers:
        #  * http.server.HTTPServer
        #  * http.server.ThreadingHTTPServer  # New in version 3.7.
        #
        # Handlers:
        #  * http.server.BaseHTTPRequestHandler
        #  * http.server.SimpleHTTPRequestHandler
        #  * http.server.CGIHTTPRequestHandler
        super().__init__()
        self._server_class = server_class
        self._handler_class = handler_class
        self._port = 8080
        self._httpd = None
        self._thread = None


    def serve(self):
        # with server_class(("", server_address), handler_class) as httpd:
        if self._httpd:
            raise RuntimeError("Server is already running")
        
        self._httpd = self._server_class(("", self._port), self._handler_class)
        import threading
        self._thread = threading.Thread(target=self._httpd.serve_forever)
        self._thread.start()
        logger.info(f"Started HTTP server at {self.url()}")

    def url(self):
        """The URL of the server.
        """
        return f"http://localhost:{self._port}/" if self._httpd else None

    def stop(self):
        if not self._httpd:
            raise RuntimeError("No server is running")

        try:
            # Tell the serve_forever() loop to stop and wait until it does
            self._httpd.shutdown()
            self._thread.join()
        finally:
            del self._httpd
            self._httpd = None
            self._thread = None
            logger.info(f"HTTP server is now down.")

