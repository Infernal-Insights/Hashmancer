#!/bin/bash

echo "🚀 Hashmancer Vast.ai Setup Script"
echo "=================================="

# Check if API key is provided
if [ -z "$VAST_API_KEY" ]; then
    echo "❌ VAST_API_KEY environment variable not set!"
    echo ""
    echo "📋 Setup Instructions:"
    echo "1. Go to https://cloud.vast.ai/"
    echo "2. Login to your account"
    echo "3. Navigate to Account → API Keys"
    echo "4. Create a new API key or copy existing one"
    echo "5. Export it: export VAST_API_KEY='your_key_here'"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ VAST_API_KEY found: ${VAST_API_KEY:0:10}..."

# Create required directories
echo "📁 Creating directory structure..."
mkdir -p server/vast_templates/wordlists
mkdir -p server/vast_templates/rules
mkdir -p server/logs
mkdir -p server/static/uploads

# Set permissions
chmod +x server/vast_templates/*.sh

# Install Python dependencies for vast.ai integration
echo "📦 Installing Python dependencies..."
pip install aiohttp tabulate click

# Test vast.ai API connection
echo "🔗 Testing vast.ai API connection..."
python3 -c "
import asyncio
import aiohttp
import os

async def test_connection():
    api_key = os.getenv('VAST_API_KEY')
    headers = {'Authorization': f'Bearer {api_key}'}
    
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get('https://console.vast.ai/api/v0/instances') as response:
                if response.status == 200:
                    print('✅ API connection successful!')
                    data = await response.json()
                    print(f'📊 Current instances: {len(data)}')
                else:
                    print(f'❌ API connection failed: {response.status}')
    except Exception as e:
        print(f'❌ Connection error: {e}')

asyncio.run(test_connection())
"

# Create environment file
echo "⚙️ Creating environment configuration..."
cat > .env << EOF
# Vast.ai Configuration
VAST_API_KEY=${VAST_API_KEY}
HASHMANCER_SERVER_URL=http://$(curl -s ifconfig.me):8080
HASHMANCER_API_KEY=$(openssl rand -hex 32)

# Portal Configuration
PORTAL_SECRET_KEY=$(openssl rand -hex 32)
PORTAL_HOST=0.0.0.0
PORTAL_PORT=8080

# Performance Settings
DEFAULT_GPU_TYPE=rtx4090
DEFAULT_MAX_PRICE=1.50
AUTO_SCALING_ENABLED=true
COST_MONITORING_ENABLED=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=server/logs/hashmancer.log
EOF

# Create systemd service (optional)
if [ "$EUID" -eq 0 ]; then
    echo "🔧 Creating systemd service..."
    cat > /etc/systemd/system/hashmancer-portal.service << EOF
[Unit]
Description=Hashmancer Worker Portal
After=network.target

[Service]
Type=simple
User=$(logname)
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
EnvironmentFile=$(pwd)/.env
ExecStart=$(pwd)/venv/bin/python server/worker_portal.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    echo "✅ Systemd service created. Start with: sudo systemctl start hashmancer-portal"
fi

# Create launch script
echo "📋 Creating quick launch script..."
cat > launch_portal.sh << 'EOF'
#!/bin/bash
set -e

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check API key
if [ -z "$VAST_API_KEY" ]; then
    echo "❌ Please set VAST_API_KEY in .env file"
    exit 1
fi

echo "🚀 Starting Hashmancer Portal..."
echo "Portal URL: http://localhost:${PORTAL_PORT:-8080}"
echo "API Key: ${VAST_API_KEY:0:10}..."

# Start the portal
cd server
python worker_portal.py
EOF
chmod +x launch_portal.sh

# Create CLI wrapper
echo "🔧 Creating CLI wrapper..."
cat > hashmancer-cli << 'EOF'
#!/bin/bash
# Hashmancer CLI wrapper script

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run CLI with proper environment
cd server
python cli_commands.py "$@"
EOF
chmod +x hashmancer-cli

# Download sample wordlists (optional)
echo "📚 Would you like to download sample wordlists? (y/n)"
read -r download_wordlists

if [ "$download_wordlists" = "y" ] || [ "$download_wordlists" = "Y" ]; then
    echo "📥 Downloading sample wordlists..."
    cd server/vast_templates/wordlists
    
    # Download RockYou (most common)
    if [ ! -f "rockyou.txt" ]; then
        echo "Downloading RockYou..."
        wget -q --show-progress "https://github.com/danielmiessler/SecLists/raw/master/Passwords/Leaked-Databases/rockyou.txt.tar.gz"
        tar -xzf rockyou.txt.tar.gz
        rm rockyou.txt.tar.gz
        echo "✅ RockYou downloaded ($(wc -l < rockyou.txt | tr -d ' ') passwords)"
    fi
    
    # Download common passwords
    if [ ! -f "common-10k.txt" ]; then
        echo "Downloading common passwords..."
        wget -q --show-progress -O common-10k.txt "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/10-million-password-list-top-10000.txt"
        echo "✅ Common passwords downloaded ($(wc -l < common-10k.txt | tr -d ' ') passwords)"
    fi
    
    cd ../../..
fi

# Create Docker image (optional)
echo "🐳 Would you like to build the Docker image for faster deployment? (y/n)"
read -r build_docker

if [ "$build_docker" = "y" ] || [ "$build_docker" = "Y" ]; then
    echo "🔨 Building Hashmancer Docker image..."
    
    # Copy requirements if not exists
    if [ ! -f requirements.txt ]; then
        cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
jinja2==3.1.2
python-multipart==0.0.6
aiohttp==3.9.1
tabulate==0.9.0
click==8.1.7
pydantic==2.5.0
python-jose==3.3.0
passlib==1.7.4
EOF
    fi
    
    # Build image
    docker build -f server/vast_templates/hashmancer_base.dockerfile -t hashmancer:latest .
    echo "✅ Docker image built: hashmancer:latest"
    
    # Optional: Push to registry
    echo "🌐 Would you like to push to Docker Hub? (requires login) (y/n)"
    read -r push_docker
    
    if [ "$push_docker" = "y" ] || [ "$push_docker" = "Y" ]; then
        echo "Please enter your Docker Hub username:"
        read -r docker_username
        
        docker tag hashmancer:latest $docker_username/hashmancer:latest
        docker push $docker_username/hashmancer:latest
        echo "✅ Image pushed to $docker_username/hashmancer:latest"
        
        # Update templates to use your image
        sed -i "s|derekhartwig/hashmancer:latest|$docker_username/hashmancer:latest|g" server/vast_templates/quick_deploy_templates.json
        echo "✅ Templates updated to use your Docker image"
    fi
fi

echo ""
echo "🎉 Hashmancer Vast.ai Setup Complete!"
echo "====================================="
echo ""
echo "📋 Quick Start Commands:"
echo "  ./launch_portal.sh           - Start the web portal"
echo "  ./hashmancer-cli workers list - List workers via CLI"
echo "  ./hashmancer-cli workers launch --gpu-type rtx4090 - Launch worker"
echo ""
echo "🌐 Web Portal:"
echo "  URL: http://localhost:8080"
echo "  Features: Dashboard, worker management, cost analysis"
echo ""
echo "💰 Cost Management:"
echo "  - Set budgets in the web portal"
echo "  - Monitor costs in real-time"
echo "  - Auto-scaling available"
echo ""
echo "📚 Documentation:"
echo "  - Templates: server/vast_templates/"
echo "  - Logs: server/logs/"
echo "  - Config: .env"
echo ""
echo "🚀 Ready to deploy workers on vast.ai!"