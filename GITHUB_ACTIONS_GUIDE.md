# 🚀 GitHub Actions Self-Hosted Runner Setup

Complete guide for setting up GitHub Actions to run on your own hardware with GPU support, NVIDIA drivers, and Hashcat integration.

## 🎯 Quick Start

### 1. Setup Self-Hosted Runner
```bash
# Run the setup script
./setup-github-runner.sh

# Follow the prompts to configure:
# - GitHub repository URL
# - Registration token (from GitHub Settings > Actions > Runners)
# - Runner name and labels
```

### 2. Test Local Workflows (Optional)
```bash
# Setup local testing with act
./setup-act.sh

# Test workflows locally
./test-workflows-local.sh
```

### 3. Manage Your Runner
```bash
# Launch interactive dashboard
./github-runner-manager.sh

# Or use specific commands
./github-runner-manager.sh status
./github-runner-manager.sh health
```

## 📋 Complete Setup Process

### Step 1: Get GitHub Registration Token

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Actions** → **Runners**
3. Click **"New self-hosted runner"**
4. Select **Linux** as the operating system
5. Copy the token from the `./config.sh` command (it looks like `AXXXXXXXXXXXXXXXXXXXXX`)

### Step 2: Run Setup Script

```bash
# Make sure you're in the hashmancer directory
cd /path/to/hashmancer

# Run the setup script
./setup-github-runner.sh
```

The script will:
- ✅ Install GitHub Actions runner
- ✅ Configure with your repository
- ✅ Set up systemd service for auto-start
- ✅ Install NVIDIA Container Toolkit (if GPU detected)
- ✅ Configure monitoring and logging
- ✅ Create management scripts

### Step 3: Verify Setup

```bash
# Check runner status
./github-runner-manager.sh status

# Run health check
./github-runner-manager.sh health

# View in GitHub
# Go to: https://github.com/your-username/hashmancer/settings/actions/runners
```

## 🎮 Workflow Configuration

Your workflows are already configured to use the self-hosted runner:

### Python Tests Workflow
- **File**: `.github/workflows/python-tests.yml`
- **Triggers**: Push, Pull Request, Release
- **Runner**: `runs-on: [self-hosted, linux, gpu]`
- **Features**: Code quality, Redis tests, Docker validation, GPU tests

### GPU Tests Workflow  
- **File**: `.github/workflows/gpu-tests.yml`
- **Triggers**: Push (docker changes), PR, Manual dispatch
- **Runner**: `runs-on: [self-hosted, linux, gpu, docker]`
- **Features**: GPU validation, Docker testing, Performance tests

### Test Levels
```yaml
# Manual trigger with test level selection
workflow_dispatch:
  inputs:
    test_level:
      description: 'Test level (basic, full, stress)'
      required: false
      default: 'basic'
      type: choice
      options:
        - basic    # Quick validation tests
        - full     # Complete deployment testing
        - stress   # Load and stress testing
```

## 🛠️ Management Commands

### Runner Manager Dashboard
```bash
# Interactive dashboard with all options
./github-runner-manager.sh
```

### Individual Commands
```bash
# Status and information
./github-runner-manager.sh status
./github-runner-manager.sh health
./github-runner-manager.sh logs

# Service control
./github-runner-manager.sh start
./github-runner-manager.sh stop
./github-runner-manager.sh restart

# Maintenance
./github-runner-manager.sh update
./github-runner-manager.sh cleanup
./github-runner-manager.sh monitor 300  # 5 minute monitoring
```

### Original Setup Scripts
```bash
# Runner setup (from $HOME/github-runner)
$HOME/github-runner/runner-status.sh
$HOME/github-runner/restart-runner.sh
$HOME/github-runner/runner-logs.sh
```

## 🧪 Local Workflow Testing

Test your workflows locally before pushing to GitHub:

### Setup act (One-time)
```bash
./setup-act.sh
```

### Test Workflows
```bash
# Interactive testing
./test-workflows-local.sh

# Specific tests
./test-workflows-local.sh python   # Test Python workflow
./test-workflows-local.sh gpu      # Test GPU workflow
./test-workflows-local.sh list     # List all workflows

# Validate syntax
./validate-workflows.sh
```

### Manual act Commands
```bash
# List workflows
act -l

# Run specific workflow
act -W .github/workflows/python-tests.yml

# Run with custom input
act -W .github/workflows/gpu-tests.yml workflow_dispatch --input test_level=full

# Dry run (validation only)
act --dry-run
```

## 🔧 Configuration Files

### Runner Configuration
- **Location**: `$HOME/github-runner/.runner`
- **Service**: Auto-installed as systemd service
- **Logs**: `/var/log/github-runner/`
- **Monitoring**: Cron job every 5 minutes

### act Configuration
- **Config**: `.actrc` (created by setup-act.sh)
- **Secrets**: `.secrets.env` (copy from template)
- **Events**: `.github/events/` (for testing)

### Workflow Files
- **Python Tests**: `.github/workflows/python-tests.yml`
- **GPU Tests**: `.github/workflows/gpu-tests.yml`

## 🎯 Runner Labels and Targeting

Your runner is configured with these labels:
- `self-hosted` - Indicates self-hosted runner
- `linux` - Linux operating system
- `x64` - 64-bit architecture
- `gpu` - NVIDIA GPU available (if detected)
- `cuda` - CUDA support available (if detected)
- `docker` - Docker available (if detected)

### Targeting in Workflows
```yaml
# Target specific capabilities
runs-on: [self-hosted, linux, gpu]          # GPU required
runs-on: [self-hosted, linux, docker]       # Docker required  
runs-on: [self-hosted, linux, gpu, docker]  # Both required
```

## 📊 Monitoring and Diagnostics

### Automated Monitoring
- **Cron Job**: Runs every 5 minutes
- **Checks**: Service status, disk space, GPU access, Docker access
- **Logs**: `/var/log/github-runner/monitor.log`
- **Auto-restart**: Attempts to restart failed services

### Manual Monitoring
```bash
# Real-time performance monitoring
./github-runner-manager.sh monitor 600  # 10 minutes

# Health check
./github-runner-manager.sh health

# View logs
./github-runner-manager.sh logs 100  # Last 100 lines

# System status
./github-runner-manager.sh status
```

### Log Locations
```bash
# Service logs (systemd)
sudo journalctl -u actions.runner.* -f

# Monitor logs
tail -f /var/log/github-runner/monitor.log

# Runner logs (if available)
tail -f $HOME/github-runner/_diag/
```

## 🔒 Security Considerations

### Runner Security
- ✅ Runs as non-root user
- ✅ Isolated work directory (`_work`)
- ✅ Network isolation per job
- ✅ Secret masking in logs
- ✅ Automatic cleanup after jobs

### System Security
- ✅ Docker socket access (required for Docker tests)
- ✅ GPU device access (required for GPU tests)
- ✅ Limited sudo access (systemctl only)
- ⚠️ Monitor for privilege escalation

### Best Practices
1. **Regular Updates**: Update runner monthly
2. **Monitor Logs**: Check for suspicious activity
3. **Limit Repository Access**: Only trusted repositories
4. **Network Security**: Firewall rules if needed
5. **Backup Configuration**: Keep runner config backed up

## 🚨 Troubleshooting

### Common Issues

#### Runner Not Showing in GitHub
```bash
# Check service status
./github-runner-manager.sh status

# Check configuration
cat $HOME/github-runner/.runner

# Re-register runner
cd $HOME/github-runner
./config.sh remove --token YOUR_TOKEN
./config.sh --url REPO_URL --token NEW_TOKEN --name RUNNER_NAME
```

#### Workflows Failing
```bash
# Check runner logs
./github-runner-manager.sh logs

# Test locally first
./test-workflows-local.sh python

# Verify prerequisites
./github-runner-manager.sh health
```

#### GPU Tests Failing
```bash
# Check GPU access
nvidia-smi

# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Verify container toolkit
nvidia-container-runtime --version
```

#### Docker Issues
```bash
# Check Docker access
docker ps

# Add user to docker group (if needed)
sudo usermod -aG docker $USER
# Then logout and login

# Restart Docker service
sudo systemctl restart docker
```

### Recovery Procedures

#### Restart Everything
```bash
# Stop runner
./github-runner-manager.sh stop

# Restart Docker
sudo systemctl restart docker

# Start runner
./github-runner-manager.sh start
```

#### Complete Reset
```bash
# Stop and remove service
cd $HOME/github-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall

# Remove configuration
./config.sh remove --token YOUR_TOKEN

# Re-run setup
./setup-github-runner.sh
```

#### Update Runner
```bash
# Automated update
./github-runner-manager.sh update

# Manual update
cd $HOME/github-runner
sudo ./svc.sh stop
# Download and extract new version
sudo ./svc.sh start
```

## 📚 Additional Resources

### Documentation
- **GitHub Actions**: https://docs.github.com/en/actions
- **Self-hosted Runners**: https://docs.github.com/en/actions/hosting-your-own-runners
- **act Tool**: https://github.com/nektos/act
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/

### Files Created
- `setup-github-runner.sh` - Main setup script
- `github-runner-manager.sh` - Management dashboard
- `setup-act.sh` - Local testing setup
- `test-workflows-local.sh` - Local workflow testing
- `validate-workflows.sh` - Workflow validation
- `ACT_USAGE.md` - act usage documentation

### Service Management
```bash
# Service name pattern
actions.runner.{repo-name}.{runner-name}

# Direct systemctl commands
sudo systemctl status actions.runner.*
sudo systemctl restart actions.runner.*
sudo systemctl enable actions.runner.*
```

## 🎉 Success Indicators

Your setup is working correctly when:

- ✅ Runner appears in GitHub Settings > Actions > Runners
- ✅ Runner shows "Online" status in GitHub  
- ✅ Health check passes: `./github-runner-manager.sh health`
- ✅ Test workflows run successfully
- ✅ GPU tests pass (if GPU available)
- ✅ Docker tests pass
- ✅ No errors in runner logs

## 💡 Tips and Best Practices

### Performance Optimization
1. **SSD Storage**: Use SSD for runner work directory
2. **Adequate RAM**: 8GB+ recommended for Docker builds
3. **Network**: Stable internet connection for downloads
4. **GPU Memory**: 4GB+ GPU memory for ML workloads

### Maintenance Schedule
- **Daily**: Check runner status in GitHub
- **Weekly**: Review logs for errors
- **Monthly**: Update runner to latest version
- **Quarterly**: Full health check and cleanup

### Scaling
- **Multiple Runners**: Run setup script on additional machines
- **Different Labels**: Use labels to target specific hardware
- **Load Balancing**: GitHub automatically distributes jobs

---

## 🎯 Quick Reference

```bash
# Essential commands
./setup-github-runner.sh           # Initial setup
./github-runner-manager.sh         # Management dashboard
./github-runner-manager.sh health  # Health check
./test-workflows-local.sh          # Local testing

# Check runner in GitHub
https://github.com/USERNAME/REPO/settings/actions/runners

# Monitor logs
tail -f /var/log/github-runner/monitor.log
```

Your GitHub Actions self-hosted runner is now ready for production use! 🚀