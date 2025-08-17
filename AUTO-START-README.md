# 🚀 Hashmancer Auto-Start Setup

This directory contains everything needed to make Hashmancer start automatically and stay running without any user intervention.

## 📋 Quick Setup

**Run this ONE command as root to set up everything:**

```bash
sudo ./install-service.sh
```

That's it! After running this command, Hashmancer will:
- ✅ Start automatically on boot
- ✅ Restart automatically if it crashes
- ✅ Monitor itself for health issues
- ✅ Handle network dependencies
- ✅ Manage logs automatically

## 🎯 What Gets Installed

### 1. **Systemd Service** (`hashmancer.service`)
- Starts Hashmancer automatically on boot
- Manages the service lifecycle
- Handles dependencies and restarts

### 2. **Startup Scripts** (`scripts/`)
- `start-hashmancer.sh` - Main startup script with error handling
- `pre-start-check.sh` - Pre-flight checks before startup
- `backup-startup.sh` - Alternative startup methods if systemd fails
- `monitor-hashmancer.sh` - Advanced health monitoring
- `health-check.sh` - Simple health checks

### 3. **Maintenance Scripts** (`scripts/`)
- `restart-hashmancer.sh` - Restart the service
- `stop-hashmancer.sh` - Stop the service
- `status-hashmancer.sh` - Check service status

### 4. **Automatic Monitoring**
- Health checks every 2 minutes via cron
- Resource monitoring (CPU, memory, disk)
- Error detection and alerting
- Automatic log rotation

## 🖥️ Portal Access

After installation, the portal will be available at:
- **Main Portal**: http://localhost:8000
- **Admin Panel**: http://localhost:8000/admin

## 📊 Monitoring & Status

### Check Service Status
```bash
# Quick status check
./scripts/status-hashmancer.sh

# Detailed monitoring report
./scripts/monitor-hashmancer.sh report

# Check systemd service
systemctl status hashmancer
```

### View Logs
```bash
# Live logs
journalctl -u hashmancer -f

# Recent logs
journalctl -u hashmancer -n 50

# Application logs
tail -f logs/startup.log
```

## 🔧 Manual Control

### Service Management
```bash
# Start service
sudo systemctl start hashmancer

# Stop service
sudo systemctl stop hashmancer

# Restart service
sudo systemctl restart hashmancer

# Enable/disable auto-start
sudo systemctl enable hashmancer   # Enable auto-start
sudo systemctl disable hashmancer  # Disable auto-start
```

### Using Scripts
```bash
# Restart using script
./scripts/restart-hashmancer.sh

# Stop using script
./scripts/stop-hashmancer.sh

# Use backup startup methods
./scripts/backup-startup.sh start
```

## 🛠️ Troubleshooting

### Portal Not Accessible

1. **Check service status:**
   ```bash
   ./scripts/status-hashmancer.sh
   ```

2. **Check logs for errors:**
   ```bash
   journalctl -u hashmancer -n 50
   tail -f logs/startup.log
   ```

3. **Try manual restart:**
   ```bash
   ./scripts/restart-hashmancer.sh
   ```

4. **Use backup startup methods:**
   ```bash
   ./scripts/backup-startup.sh start
   ```

### Service Won't Start

1. **Check system dependencies:**
   ```bash
   # Install missing dependencies
   sudo apt update
   sudo apt install python3 python3-pip python3-venv redis-server
   ```

2. **Check permissions:**
   ```bash
   # Fix script permissions
   chmod +x scripts/*.sh
   ```

3. **Check disk space:**
   ```bash
   df -h /home/derekhartwig/hashmancer
   ```

4. **Manual startup test:**
   ```bash
   cd /home/derekhartwig/hashmancer
   python3 -m hashmancer.server.main
   ```

### High Resource Usage

1. **Check resource status:**
   ```bash
   ./scripts/monitor-hashmancer.sh check
   ```

2. **View detailed report:**
   ```bash
   ./scripts/monitor-hashmancer.sh report
   ```

3. **Clean up logs:**
   ```bash
   ./scripts/monitor-hashmancer.sh cleanup
   ```

## 📁 File Structure

```
hashmancer/
├── hashmancer.service          # Systemd service file
├── install-service.sh          # One-time installation script
├── AUTO-START-README.md        # This file
├── scripts/
│   ├── start-hashmancer.sh     # Main startup script
│   ├── pre-start-check.sh      # Pre-flight checks
│   ├── backup-startup.sh       # Backup startup methods
│   ├── monitor-hashmancer.sh   # Health monitoring
│   ├── health-check.sh         # Simple health check
│   ├── restart-hashmancer.sh   # Restart script
│   ├── stop-hashmancer.sh      # Stop script
│   └── status-hashmancer.sh    # Status script
└── logs/
    ├── startup.log             # Startup logs
    ├── monitor.log             # Monitoring logs
    ├── health-check.log        # Health check logs
    └── backup-startup.log      # Backup startup logs
```

## ⚙️ Advanced Configuration

### Email Alerts
To enable email alerts, edit `scripts/monitor-hashmancer.sh`:
```bash
ALERT_EMAIL="admin@example.com"
```

### Webhook Alerts
To enable webhook alerts (Slack, Discord, etc.):
```bash
WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

### Monitoring Thresholds
Adjust monitoring thresholds in `scripts/monitor-hashmancer.sh`:
```bash
MAX_CPU_PERCENT=90        # CPU usage threshold
MAX_MEMORY_PERCENT=85     # Memory usage threshold
MAX_DISK_PERCENT=90       # Disk usage threshold
MIN_FREE_DISK_GB=5        # Minimum free disk space
MAX_RESPONSE_TIME=10      # Portal response time (seconds)
```

## 🔄 Backup Startup Methods

If the systemd service fails, the system will automatically try these methods:

1. **Direct Python** - Direct execution of the Python module
2. **Nohup** - Background process with nohup
3. **Screen** - Detached screen session
4. **Tmux** - Detached tmux session
5. **Docker** - Container-based startup (if available)

## 📋 Log Management

- **Automatic Rotation**: Logs are rotated daily, keeping 30 days
- **Size Limits**: Large logs (>100MB) are automatically truncated
- **Cleanup**: Old logs are cleaned up automatically
- **Locations**: All logs are stored in the `logs/` directory

## 🔒 Security Notes

- Service runs as root to ensure hardware access
- Firewall rules are configured for port 8000
- SSH access is preserved during firewall setup
- No sensitive data is logged

## 🆘 Emergency Procedures

### Complete Reset
```bash
# Stop everything
sudo systemctl stop hashmancer
./scripts/backup-startup.sh stop

# Reinstall service
sudo ./install-service.sh

# Start fresh
sudo systemctl start hashmancer
```

### Manual Recovery
```bash
# If all else fails, start manually
cd /home/derekhartwig/hashmancer
source venv/bin/activate
python3 -m hashmancer.server.main
```

## 📞 Support

If you encounter issues:

1. Check this README for troubleshooting steps
2. Review the logs in the `logs/` directory
3. Run the status and monitoring scripts
4. Try the backup startup methods

The system is designed to be extremely robust and self-healing. Multiple layers of redundancy ensure the portal stays available even if individual components fail.