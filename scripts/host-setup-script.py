#!/usr/bin/env python3
"""
Host the Vast.ai worker setup script on a simple HTTP server
This allows the setup script to be downloaded via wget/curl
"""

import http.server
import socketserver
from pathlib import Path
import sys

class SetupScriptHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.script_dir = Path(__file__).parent
        super().__init__(*args, directory=str(self.script_dir), **kwargs)
    
    def do_GET(self):
        if self.path == '/vast-worker-setup.sh' or self.path == '/setup':
            # Serve the setup script
            script_file = self.script_dir / 'vast-worker-setup.sh'
            if script_file.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.send_header('Content-Disposition', 'attachment; filename="vast-worker-setup.sh"')
                self.end_headers()
                
                with open(script_file, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Setup script not found')
        else:
            # Default behavior for other files
            super().do_GET()

def start_server(port=8888):
    """Start the setup script hosting server"""
    with socketserver.TCPServer(("", port), SetupScriptHandler) as httpd:
        print(f"🌐 Hosting setup script at http://localhost:{port}/vast-worker-setup.sh")
        print(f"📋 Use: wget -O - http://YOUR_IP:{port}/setup | bash")
        print("Press Ctrl+C to stop...")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    start_server(port)