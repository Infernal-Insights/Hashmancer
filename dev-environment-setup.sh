#!/bin/bash

# Hashmancer Development Environment Setup Script
# Creates a comprehensive development and testing environment

set -e

echo "=============================================="
echo "Hashmancer Development Environment Setup"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root for security reasons"
   exit 1
fi

# System information
log_info "System information:"
echo "  OS: $(lsb_release -d | cut -f2)"
echo "  Kernel: $(uname -r)"
echo "  CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | sed 's/^/  /'
else
    log_warning "No NVIDIA GPU detected. Some features will be limited."
fi

echo

# 1. Install system dependencies
log_info "Installing system dependencies..."

# Update package lists
sudo apt update

# Essential development tools
sudo apt install -y \
    build-essential cmake ninja-build \
    git git-lfs curl wget \
    python3-dev python3-pip python3-venv \
    nodejs npm \
    htop btop nvtop \
    jq yq \
    tree fd-find ripgrep \
    tmux screen \
    docker.io docker-compose \
    postgresql-client \
    redis-tools \
    apache2-utils wrk \
    linux-tools-generic \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

log_success "System dependencies installed"

# 2. Install CUDA if not present
if ! command -v nvcc &> /dev/null; then
    log_info "Installing CUDA Toolkit..."
    
    # Add NVIDIA package repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d '.')/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt update
    
    # Install CUDA
    sudo apt install -y cuda-toolkit-12-3
    
    # Add to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    log_success "CUDA Toolkit installed"
else
    log_success "CUDA already available: $(nvcc --version | grep release)"
fi

# 3. Install Docker GPU support
if command -v docker &> /dev/null && command -v nvidia-smi &> /dev/null; then
    log_info "Installing Docker GPU support..."
    
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    log_success "Docker GPU support installed"
fi

# 4. Setup Python development environment
log_info "Setting up Python development environment..."

# Create virtual environment
python3 -m venv ~/.venv/hashmancer
source ~/.venv/hashmancer/bin/activate

# Install Python packages
pip install --upgrade pip setuptools wheel

# Core development packages
pip install \
    fastapi uvicorn \
    sqlalchemy alembic \
    redis celery \
    pytest pytest-asyncio pytest-cov \
    black isort flake8 mypy \
    jupyter jupyterlab \
    pandas numpy matplotlib seaborn \
    psutil gpustat \
    aiohttp requests \
    pydantic \
    python-multipart \
    python-jose[cryptography] \
    passlib[bcrypt] \
    prometheus-client

# Performance profiling
pip install py-spy memory-profiler line-profiler cProfile-viewer

# Add activation to bashrc
echo "alias activate-hashmancer='source ~/.venv/hashmancer/bin/activate'" >> ~/.bashrc

log_success "Python environment setup complete"

# 5. Setup Node.js environment (for web interface)
log_info "Setting up Node.js environment..."

# Install Node Version Manager
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install latest LTS Node.js
nvm install --lts
nvm use --lts

# Install global packages
npm install -g \
    typescript \
    @types/node \
    nodemon \
    pm2 \
    create-react-app \
    @vue/cli \
    webpack webpack-cli

log_success "Node.js environment setup complete"

# 6. Create development directories
log_info "Creating development structure..."

mkdir -p ~/hashmancer-dev/{
    datasets/{hashes,wordlists,rules},
    results,
    logs,
    backups,
    scripts,
    monitoring,
    notebooks,
    docs
}

# Download common datasets
cd ~/hashmancer-dev/datasets

# Wordlists
if [ ! -d "wordlists/rockyou" ]; then
    log_info "Downloading common wordlists..."
    mkdir -p wordlists
    cd wordlists
    
    # RockYou (if available)
    wget -q --show-progress https://github.com/brannondorsey/naive-hashcat/releases/download/data/rockyou.txt || \
    log_warning "RockYou wordlist not available via direct download"
    
    # Create sample wordlists for testing
    cat > sample_passwords.txt << 'EOF'
password
123456
password123
admin
qwerty
letmein
welcome
monkey
dragon
master
hello
freedom
whatever
qazwsx
trustno1
EOF
    
    cd ..
fi

# Rules
if [ ! -d "rules" ]; then
    log_info "Downloading rule sets..."
    mkdir -p rules
    cd rules
    
    # Best64 rules
    cat > best64.rule << 'EOF'
:
l
u
c
C
t
TN
r
d
f
{
}
$!
$@
$#
$$
$%
$^
$&
$*
$(
$)
$-
$_
$+
$=
$1
$2
$3
$4
$5
$6
$7
$8
$9
$0
^!
^@
^#
^$
^%
^^
^&
^*
^(
^)
^-
^_
^+
^=
^1
^2
^3
^4
^5
^6
^7
^8
^9
^0
se3
sa@
ss$
si1
so0
EOF

    # Common rules
    cat > common.rule << 'EOF'
:
l
u
c
d
r
$1
$2
$3
$!
^1
^2
^!
se3
sa@
so0
si1
EOF

    cd ..
fi

# Sample hashes
if [ ! -f "hashes/test_hashes.txt" ]; then
    log_info "Creating sample hash files..."
    mkdir -p hashes
    
    # MD5 test hashes
    cat > hashes/md5_test.txt << 'EOF'
5d41402abc4b2a76b9719d911017c592:hello
098f6bcd4621d373cade4e832627b4f6:test
e99a18c428cb38d5f260853678922e03:abc123
5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8:password
EOF

    # SHA1 test hashes  
    cat > hashes/sha1_test.txt << 'EOF'
aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d:hello
a94a8fe5ccb19ba61c4c0873d391e987982fbbd3:test
6367c48dd193d56ea7b0baad25b19455e529f5ee:abc123
5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8:password
EOF
fi

cd ~/hashmancer-dev

log_success "Development structure created"

# 7. Setup monitoring configuration
log_info "Setting up monitoring configuration..."

mkdir -p monitoring/{prometheus,grafana}

# Prometheus configuration
cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hashmancer-server'
    static_configs:
      - targets: ['hashmancer-server:9090']
  
  - job_name: 'hashmancer-workers'
    static_configs:
      - targets: ['hashmancer-worker:9091']
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
  
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9445']
EOF

# Grafana dashboard for GPU monitoring
mkdir -p monitoring/grafana/dashboards
cat > monitoring/grafana/dashboards/gpu-monitoring.json << 'EOF'
{
  "dashboard": {
    "title": "Hashmancer GPU Monitoring",
    "tags": ["hashmancer", "gpu"],
    "panels": [
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_ml_py_utilization_gpu",
            "legendFormat": "GPU {{device}}"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "graph", 
        "targets": [
          {
            "expr": "nvidia_ml_py_memory_used_bytes / nvidia_ml_py_memory_total_bytes * 100",
            "legendFormat": "GPU {{device}} Memory %"
          }
        ]
      },
      {
        "title": "Hash Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "hashmancer_hash_rate_total",
            "legendFormat": "H/s"
          }
        ]
      }
    ]
  }
}
EOF

log_success "Monitoring configuration created"

# 8. Create development scripts
log_info "Creating development scripts..."

mkdir -p scripts

# Build script
cat > scripts/build-all.sh << 'EOF'
#!/bin/bash
set -e

echo "Building Hashmancer components..."

# Build Darkling
cd darkling
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
cd ../..

# Build Server
cd server
pip install -r requirements.txt
cd ..

# Build Worker
cd worker  
pip install -r requirements.txt
cd ..

echo "Build complete!"
EOF

# Test script
cat > scripts/run-tests.sh << 'EOF'
#!/bin/bash
set -e

echo "Running Hashmancer test suite..."

# Test Darkling
cd darkling
./gpu_test_suite.sh
cd ..

# Test Server
cd server
python -m pytest tests/ -v
cd ..

# Test Worker
cd worker
python -m pytest tests/ -v  
cd ..

# Integration tests
python -m pytest integration_tests/ -v

echo "All tests completed!"
EOF

# Development startup script
cat > scripts/dev-start.sh << 'EOF'
#!/bin/bash

echo "Starting Hashmancer development environment..."

# Start infrastructure
docker-compose -f docker-compose.dev.yml up -d postgres redis prometheus grafana

# Wait for services
sleep 10

# Start server in development mode
cd server && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080 &

# Start worker simulator
cd worker && python worker_simulator.py &

echo "Development environment running!"
echo "Server: http://localhost:8080"
echo "Grafana: http://localhost:3000 (admin/admin123)"
echo "Jupyter: http://localhost:8888"
EOF

# Make scripts executable
chmod +x scripts/*.sh

log_success "Development scripts created"

# 9. Setup IDE configuration
log_info "Setting up IDE configuration..."

# VS Code settings
if command -v code &> /dev/null; then
    mkdir -p .vscode
    
    cat > .vscode/settings.json << 'EOF'
{
    "python.pythonPath": "~/.venv/hashmancer/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "cmake.configureOnOpen": true,
    "files.associations": {
        "*.cu": "cuda-cpp",
        "*.cuh": "cuda-cpp"
    }
}
EOF

    cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-vscode.cpptools",
        "ms-vscode.cmake-tools",
        "nvidia.nsight-vscode-edition",
        "ms-vscode.docker",
        "ms-kubernetes-tools.vscode-kubernetes-tools",
        "redhat.vscode-yaml",
        "esbenp.prettier-vscode"
    ]
}
EOF

    log_success "VS Code configuration created"
fi

# 10. Create documentation
log_info "Creating documentation structure..."

mkdir -p docs/{api,architecture,deployment,tutorials}

cat > docs/README.md << 'EOF'
# Hashmancer Development Environment

## Quick Start
1. Run `source ~/.bashrc` to update PATH
2. Activate Python environment: `activate-hashmancer`
3. Start development environment: `./scripts/dev-start.sh`
4. Run tests: `./scripts/run-tests.sh`

## Architecture
- **Server**: FastAPI-based coordination server
- **Worker**: GPU-accelerated password cracking nodes  
- **Darkling**: High-performance CUDA engine
- **Monitoring**: Prometheus + Grafana stack

## Development Workflow
1. Make changes to code
2. Run relevant tests
3. Check monitoring for performance impact
4. Submit PR with test coverage

## Datasets Location
- Wordlists: `~/hashmancer-dev/datasets/wordlists/`
- Rules: `~/hashmancer-dev/datasets/rules/`
- Test hashes: `~/hashmancer-dev/datasets/hashes/`

## Monitoring
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9091
- Server metrics: http://localhost:8080/metrics
EOF

log_success "Documentation created"

# Final summary
echo
echo "=============================================="
log_success "Development Environment Setup Complete!"
echo "=============================================="
echo
echo "Next steps:"
echo "1. Run 'source ~/.bashrc' to update your PATH"
echo "2. Activate Python environment: 'activate-hashmancer'"
echo "3. Navigate to your Hashmancer code directory"
echo "4. Start development: './scripts/dev-start.sh'"
echo
echo "Key locations:"
echo "  Development data: ~/hashmancer-dev/"
echo "  Python env: ~/.venv/hashmancer/"
echo "  Scripts: ./scripts/"
echo "  Monitoring: http://localhost:3000"
echo
echo "For GPU testing, run:"
echo "  cd darkling && ./gpu_test_suite.sh"
echo
log_success "Happy hacking! 🚀"