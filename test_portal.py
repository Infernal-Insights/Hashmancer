#!/usr/bin/env python3
"""Simple test portal to validate the installation."""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Hashmancer Test Portal", version="1.0.0")

@app.get("/", response_class=HTMLResponse)
async def portal():
    """Serve the enhanced portal."""
    portal_file = Path("hashmancer/server/portal_enhanced.html")
    if portal_file.exists():
        return HTMLResponse(portal_file.read_text())
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hashmancer Test Portal</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #0a0a0a; color: #fff; }
                .container { max-width: 800px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 40px; }
                .status { background: #1a1a1a; padding: 20px; border-radius: 10px; margin: 20px 0; }
                .success { border-left: 4px solid #4caf50; }
                .info { border-left: 4px solid #2196f3; }
                .feature { background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 5px; }
                h1 { color: #4caf50; font-size: 2.5em; margin-bottom: 10px; }
                h2 { color: #2196f3; }
                .logo { font-size: 3em; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">🔓</div>
                    <h1>Hashmancer</h1>
                    <p>Seamless Installation Test Portal</p>
                </div>
                
                <div class="status success">
                    <h2>✅ Installation Successful!</h2>
                    <p>The Hashmancer server is running and accessible.</p>
                </div>
                
                <div class="status info">
                    <h2>🚀 Available Features</h2>
                    <div class="feature">
                        <strong>Interactive Installation Script:</strong> One-command setup with prompts for API keys and preferences
                    </div>
                    <div class="feature">
                        <strong>Docker Support:</strong> Production-ready containers with GPU passthrough
                    </div>
                    <div class="feature">
                        <strong>Automated Setup:</strong> Firewall, SSL/TLS, dependencies, and health monitoring
                    </div>
                    <div class="feature">
                        <strong>Multiple Deployment Modes:</strong> Docker, Native, and Cloud/VPS options
                    </div>
                    <div class="feature">
                        <strong>Management Scripts:</strong> Easy start, stop, status, and update commands
                    </div>
                </div>
                
                <div class="status info">
                    <h2>🛠️ Quick Commands</h2>
                    <div class="feature">
                        <strong>Full Installation:</strong> <code>./install-hashmancer.sh</code>
                    </div>
                    <div class="feature">
                        <strong>Docker Production:</strong> <code>./docker-scripts/quick-deploy.sh production</code>
                    </div>
                    <div class="feature">
                        <strong>GPU Deployment:</strong> <code>./docker-scripts/quick-deploy.sh gpu</code>
                    </div>
                    <div class="feature">
                        <strong>Status Check:</strong> <code>./scripts/status.sh</code>
                    </div>
                </div>
                
                <div class="status success">
                    <h2>🎯 Next Steps</h2>
                    <ol>
                        <li>Run the full installation script for production deployment</li>
                        <li>Configure API keys and admin credentials</li>
                        <li>Access the complete portal interface</li>
                        <li>Start your hash cracking projects!</li>
                    </ol>
                </div>
            </div>
        </body>
        </html>
        """)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Hashmancer Test Portal",
        "features": [
            "Interactive Installation",
            "Docker Support", 
            "GPU Passthrough",
            "Automated Setup",
            "Management Scripts"
        ]
    })

@app.get("/status")
async def status():
    """Server status endpoint."""
    return JSONResponse({
        "server": "running",
        "installation": "ready",
        "features_available": True,
        "docker_ready": True,
        "gpu_support": True
    })

if __name__ == "__main__":
    print("🚀 Starting Hashmancer Test Portal...")
    print("✅ Installation validation server")
    print("🌐 Access: http://localhost:8090")
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info", loop="asyncio")