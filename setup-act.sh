#!/bin/bash
set -e

# Setup act for local GitHub Actions testing
# This script installs and configures act to test workflows locally

echo "🎭 Setting up act for Local GitHub Actions Testing"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if act is already installed
check_act_installation() {
    if command -v act > /dev/null 2>&1; then
        local version=$(act --version 2>/dev/null | head -1 | awk '{print $3}' || echo "unknown")
        log_success "act is already installed (version: $version)"
        return 0
    else
        return 1
    fi
}

# Install act
install_act() {
    log_info "Installing act..."
    
    # Detect architecture
    local arch=$(uname -m)
    case $arch in
        x86_64)
            arch="x86_64"
            ;;
        aarch64|arm64)
            arch="arm64"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
    
    # Get latest version
    log_info "Getting latest act version..."
    local latest_version=$(curl -s https://api.github.com/repos/nektos/act/releases/latest | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')
    
    if [[ -z "$latest_version" ]]; then
        log_error "Could not determine latest act version"
        exit 1
    fi
    
    log_info "Latest act version: $latest_version"
    
    # Download and install
    local download_url="https://github.com/nektos/act/releases/download/v${latest_version}/act_Linux_${arch}.tar.gz"
    local temp_dir=$(mktemp -d)
    
    log_info "Downloading act from $download_url..."
    curl -L "$download_url" -o "$temp_dir/act.tar.gz"
    
    log_info "Extracting act..."
    cd "$temp_dir"
    tar xzf act.tar.gz
    
    log_info "Installing act to /usr/local/bin..."
    sudo mv act /usr/local/bin/
    sudo chmod +x /usr/local/bin/act
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log_success "act installed successfully"
}

# Create act configuration
create_act_config() {
    log_info "Creating act configuration..."
    
    # Create .actrc configuration file
    cat << EOF > .actrc
# act configuration for Hashmancer
# Use custom runner image with GPU support

# Platform configuration
-P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-latest
-P self-hosted=ghcr.io/catthehacker/ubuntu:act-latest

# Bind Docker socket for Docker-in-Docker
--bind /var/run/docker.sock:/var/run/docker.sock

# Set environment variables
--env DOCKER_HOST=unix:///var/run/docker.sock
--env GITHUB_ACTIONS=true

# Use local secrets if available
--secret-file .secrets.env

# Verbose output for debugging
--verbose

# Container options for GPU support (if available)
--container-options "--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm"
EOF

    log_success "Created .actrc configuration"
    
    # Create secrets template
    if [[ ! -f .secrets.env ]]; then
        cat << EOF > .secrets.env.template
# Environment variables for local testing
# Copy this to .secrets.env and fill in actual values

# GitHub token for API access (optional)
GITHUB_TOKEN=your_github_token_here

# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Custom environment variables
LOG_LEVEL=DEBUG
PYTHONPATH=.
EOF
        
        log_info "Created .secrets.env.template - copy to .secrets.env and fill in values"
    fi
    
    # Create GitHub event files for testing
    mkdir -p .github/events
    
    # Push event
    cat << EOF > .github/events/push.json
{
  "push": {
    "head_commit": {
      "id": "test123",
      "message": "Local test commit"
    }
  },
  "ref": "refs/heads/main",
  "repository": {
    "name": "hashmancer",
    "full_name": "test/hashmancer"
  }
}
EOF

    # Pull request event
    cat << EOF > .github/events/pull_request.json
{
  "action": "opened",
  "number": 1,
  "pull_request": {
    "id": 1,
    "number": 1,
    "head": {
      "ref": "feature-branch",
      "sha": "test123"
    },
    "base": {
      "ref": "main"
    }
  },
  "repository": {
    "name": "hashmancer",
    "full_name": "test/hashmancer"
  }
}
EOF

    log_success "Created GitHub event files for testing"
}

# Create test scripts
create_test_scripts() {
    log_info "Creating local testing scripts..."
    
    # Script to test Python workflow
    cat << 'EOF' > test-workflows-local.sh
#!/bin/bash
# Local GitHub Actions workflow testing script

echo "🧪 Testing GitHub Actions Workflows Locally"
echo "==========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "\033[0;34mℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Test basic Python workflow
test_python_workflow() {
    log_info "Testing Python workflow..."
    
    if act -W .github/workflows/python-tests.yml --dry-run; then
        log_success "Python workflow dry-run successful"
        
        log_info "Running Python workflow locally..."
        if act -W .github/workflows/python-tests.yml push --eventpath .github/events/push.json; then
            log_success "Python workflow executed successfully"
        else
            log_error "Python workflow execution failed"
            return 1
        fi
    else
        log_error "Python workflow dry-run failed"
        return 1
    fi
}

# Test GPU workflow
test_gpu_workflow() {
    log_info "Testing GPU workflow..."
    
    if act -W .github/workflows/gpu-tests.yml --dry-run; then
        log_success "GPU workflow dry-run successful"
        
        log_info "Running GPU workflow locally (basic level)..."
        if act -W .github/workflows/gpu-tests.yml workflow_dispatch --eventpath .github/events/push.json \
           --input test_level=basic; then
            log_success "GPU workflow executed successfully"
        else
            log_warning "GPU workflow execution failed (may be expected without GPU)"
        fi
    else
        log_error "GPU workflow dry-run failed"
        return 1
    fi
}

# List available workflows
list_workflows() {
    log_info "Available workflows:"
    act -l
}

# Interactive mode
interactive_mode() {
    echo ""
    echo "🎮 Interactive Workflow Testing"
    echo "==============================="
    echo "1. List all workflows"
    echo "2. Test Python workflow"
    echo "3. Test GPU workflow"
    echo "4. Test specific workflow"
    echo "5. Exit"
    echo ""
    
    read -p "Choose an option (1-5): " choice
    
    case $choice in
        1)
            list_workflows
            interactive_mode
            ;;
        2)
            test_python_workflow
            interactive_mode
            ;;
        3)
            test_gpu_workflow
            interactive_mode
            ;;
        4)
            echo "Available workflows:"
            ls .github/workflows/
            read -p "Enter workflow filename: " workflow
            if [[ -f ".github/workflows/$workflow" ]]; then
                act -W ".github/workflows/$workflow" push --eventpath .github/events/push.json
            else
                log_error "Workflow not found"
            fi
            interactive_mode
            ;;
        5)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            log_error "Invalid option"
            interactive_mode
            ;;
    esac
}

# Main execution
main() {
    # Check if act is available
    if ! command -v act > /dev/null 2>&1; then
        log_error "act is not installed. Run ./setup-act.sh first"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -d ".github/workflows" ]]; then
        log_error "No .github/workflows directory found. Are you in the right directory?"
        exit 1
    fi
    
    case "${1:-interactive}" in
        "python")
            test_python_workflow
            ;;
        "gpu")
            test_gpu_workflow
            ;;
        "list")
            list_workflows
            ;;
        "interactive"|"")
            interactive_mode
            ;;
        *)
            echo "Usage: $0 [python|gpu|list|interactive]"
            echo ""
            echo "Commands:"
            echo "  python      Test Python workflow"
            echo "  gpu         Test GPU workflow" 
            echo "  list        List all workflows"
            echo "  interactive Interactive mode (default)"
            ;;
    esac
}

main "$@"
EOF

    chmod +x test-workflows-local.sh
    log_success "Created test-workflows-local.sh"
    
    # Create workflow validation script
    cat << 'EOF' > validate-workflows.sh
#!/bin/bash
# Validate GitHub Actions workflows

echo "🔍 Validating GitHub Actions Workflows"
echo "======================================"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Validate all workflows
for workflow in .github/workflows/*.yml; do
    if [[ -f "$workflow" ]]; then
        echo "Validating $(basename "$workflow")..."
        
        # Check YAML syntax
        if python3 -c "import yaml; yaml.safe_load(open('$workflow'))" 2>/dev/null; then
            log_success "$(basename "$workflow") has valid YAML syntax"
        else
            log_error "$(basename "$workflow") has invalid YAML syntax"
            continue
        fi
        
        # Validate with act
        if act -W "$workflow" --dry-run > /dev/null 2>&1; then
            log_success "$(basename "$workflow") passes act validation"
        else
            log_error "$(basename "$workflow") fails act validation"
        fi
    fi
done

echo ""
echo "🎉 Workflow validation complete!"
EOF

    chmod +x validate-workflows.sh
    log_success "Created validate-workflows.sh"
}

# Setup Docker support for act
setup_docker_support() {
    log_info "Setting up Docker support for act..."
    
    # Check if Docker is available
    if ! command -v docker > /dev/null 2>&1; then
        log_warning "Docker not found - some act features will not work"
        return 0
    fi
    
    # Check if user is in docker group
    if ! groups | grep -q docker; then
        log_warning "User is not in docker group - you may need to run act with sudo"
        log_info "To add user to docker group: sudo usermod -aG docker \$USER"
    fi
    
    # Pull commonly used Docker images for act
    log_info "Pulling Docker images for act..."
    docker pull ghcr.io/catthehacker/ubuntu:act-latest || log_warning "Failed to pull act Ubuntu image"
    
    # Check for NVIDIA Docker support
    if command -v nvidia-docker > /dev/null 2>&1 || docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi > /dev/null 2>&1; then
        log_success "NVIDIA Docker support detected"
        docker pull nvidia/cuda:12.1-base || log_warning "Failed to pull CUDA base image"
    else
        log_info "No NVIDIA Docker support detected (GPU tests will be skipped)"
    fi
    
    log_success "Docker support setup complete"
}

# Create documentation
create_documentation() {
    log_info "Creating act documentation..."
    
    cat << 'EOF' > ACT_USAGE.md
# Local GitHub Actions Testing with act

This directory is configured to test GitHub Actions workflows locally using [act](https://github.com/nektos/act).

## Quick Start

1. **Setup act** (one-time):
   ```bash
   ./setup-act.sh
   ```

2. **Test workflows locally**:
   ```bash
   ./test-workflows-local.sh
   ```

3. **Validate workflow syntax**:
   ```bash
   ./validate-workflows.sh
   ```

## Available Commands

### Interactive Testing
```bash
./test-workflows-local.sh
```

### Specific Workflow Testing
```bash
# Test Python workflow
./test-workflows-local.sh python

# Test GPU workflow  
./test-workflows-local.sh gpu

# List all workflows
./test-workflows-local.sh list
```

### Manual act Commands

```bash
# List all workflows
act -l

# Run Python tests workflow
act -W .github/workflows/python-tests.yml

# Run GPU tests with specific input
act -W .github/workflows/gpu-tests.yml workflow_dispatch --input test_level=basic

# Dry run (validate without executing)
act -W .github/workflows/python-tests.yml --dry-run

# Run with custom event
act push --eventpath .github/events/push.json
```

## Configuration

### act Configuration (`.actrc`)
- Uses Ubuntu latest image for compatibility
- Binds Docker socket for Docker-in-Docker
- Sets up environment variables
- Configures GPU support (if available)

### Environment Variables (`.secrets.env`)
Copy `.secrets.env.template` to `.secrets.env` and configure:
- `GITHUB_TOKEN` - For API access (optional)
- `REDIS_HOST` - Redis server host
- `LOG_LEVEL` - Logging level

### Event Files (`.github/events/`)
- `push.json` - Simulates push events
- `pull_request.json` - Simulates PR events

## Limitations

1. **GPU Support**: Limited GPU testing in containers (hardware dependent)
2. **Self-hosted Runners**: act simulates GitHub-hosted runners by default
3. **Secrets**: Some secrets may not be available locally
4. **External Services**: Limited access to external services

## Troubleshooting

### Common Issues

1. **Docker Permission Denied**
   ```bash
   sudo usermod -aG docker $USER
   # Then logout and login
   ```

2. **GPU Tests Failing**
   - Normal if no NVIDIA GPU available
   - Check NVIDIA Docker runtime installation

3. **Workflow Validation Errors**
   ```bash
   # Check YAML syntax
   python3 -c "import yaml; yaml.safe_load(open('.github/workflows/python-tests.yml'))"
   
   # Validate with act
   act -W .github/workflows/python-tests.yml --dry-run
   ```

4. **Missing Dependencies**
   ```bash
   # Install missing packages
   sudo apt-get update
   sudo apt-get install python3-yaml curl
   ```

### Debugging

```bash
# Verbose output
act --verbose

# Debug mode
act --debug

# List available actions
act -l

# Check runner environment
act -W .github/workflows/python-tests.yml --list
```

## Integration with Development

### Pre-commit Testing
Add to your development workflow:
```bash
# Before committing
./validate-workflows.sh
./test-workflows-local.sh python
```

### IDE Integration
Many IDEs support act integration:
- VS Code: GitHub Actions extension
- IntelliJ: GitHub Actions plugin

## Resources

- [act Documentation](https://github.com/nektos/act)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
EOF

    log_success "Created ACT_USAGE.md documentation"
}

# Main setup function
main() {
    echo ""
    
    # Check if act is already installed
    if check_act_installation; then
        log_info "Proceeding with configuration..."
    else
        install_act
    fi
    
    echo ""
    create_act_config
    echo ""
    create_test_scripts
    echo ""
    setup_docker_support
    echo ""
    create_documentation
    
    echo ""
    echo "🎉 act Setup Complete!"
    echo "===================="
    echo ""
    echo "📋 What's been set up:"
    echo "  ✅ act installed and configured"
    echo "  ✅ Local testing scripts created"
    echo "  ✅ Docker support configured"
    echo "  ✅ Workflow validation tools"
    echo "  ✅ Documentation created"
    echo ""
    echo "🚀 Next Steps:"
    echo "  1. Copy .secrets.env.template to .secrets.env and configure"
    echo "  2. Test your workflows: ./test-workflows-local.sh"
    echo "  3. Validate workflows: ./validate-workflows.sh"
    echo ""
    echo "📖 Read ACT_USAGE.md for detailed usage instructions"
    echo ""
    log_success "Ready for local GitHub Actions testing!"
}

# Handle command line arguments
case "${1:-}" in
    "help"|"-h"|"--help")
        echo "act Setup Script for Hashmancer"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)    Full setup"
        echo "  help         Show this help"
        echo ""
        exit 0
        ;;
    *)
        main
        ;;
esac