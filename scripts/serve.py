import http.server
import socketserver
import os
import sys

DIRECTORY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'artifacts/rakshapay-admin/dist/public')
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5173

class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def translate_path(self, path):
        clean_path = path.split('?')[0].split('#')[0]
        words = [w for w in clean_path.split('/') if w]
        target = DIRECTORY
        for word in words:
            target = os.path.join(target, word)

        if os.path.exists(target):
            return target

        # For any non-existent route (SPA client-side navigation like /accounts/rp-138), serve index.html
        return os.path.join(DIRECTORY, 'index.html')

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), SPAHandler) as httpd:
    print(f"Serving SPA on http://localhost:{PORT} from {DIRECTORY}")
    httpd.serve_forever()
