#!/bin/bash

# Unified Hashmancer Server Setup
# Configures server as NFS + Management node + Coordination server all-in-one

set -e

echo "=============================================="
echo "Hashmancer Unified Server Setup"
echo "NFS + Management + Coordination Server"
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

# Check if running as root for system configuration
if [[ $EUID -ne 0 ]]; then
   log_error "This script needs to be run as root for system configuration"
   echo "Run: sudo ./unified-server-setup.sh"
   exit 1
fi

# Get the non-root user who will own the services
HASHMANCER_USER=${SUDO_USER:-$USER}
HASHMANCER_HOME=$(getent passwd "$HASHMANCER_USER" | cut -d: -f6)

log_info "Setting up unified server for user: $HASHMANCER_USER"
log_info "Home directory: $HASHMANCER_HOME"

# 1. Install NFS server components
log_info "Installing NFS server components..."

apt update
apt install -y \
    nfs-kernel-server \
    nfs-common \
    rpcbind \
    portmap \
    netbase

systemctl enable nfs-kernel-server
systemctl enable rpcbind

log_success "NFS server components installed"

# 2. Create shared storage structure
log_info "Creating shared storage structure..."

# Main shared directory
SHARED_ROOT="/srv/hashmancer"
mkdir -p $SHARED_ROOT/{datasets,results,logs,configs,backups,temp}

# Datasets organization
mkdir -p $SHARED_ROOT/datasets/{wordlists,rules,hashes,custom}

# Results organization  
mkdir -p $SHARED_ROOT/results/{completed,active,failed,archived}

# Logs organization
mkdir -p $SHARED_ROOT/logs/{server,workers,system,audit}

# Configuration templates
mkdir -p $SHARED_ROOT/configs/{server,worker,darkling}

# Set ownership and permissions
chown -R $HASHMANCER_USER:$HASHMANCER_USER $SHARED_ROOT
chmod -R 755 $SHARED_ROOT

# Worker-specific permissions for results and logs
chmod -R 775 $SHARED_ROOT/{results,logs,temp}

log_success "Shared storage structure created at $SHARED_ROOT"

# 3. Configure NFS exports
log_info "Configuring NFS exports..."

# Backup existing exports
cp /etc/exports /etc/exports.backup.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

# Create NFS exports configuration
cat >> /etc/exports << EOF

# Hashmancer NFS Exports
# Shared datasets (read-only for workers)
$SHARED_ROOT/datasets *(ro,sync,no_subtree_check,no_root_squash)

# Results directory (read-write for workers)
$SHARED_ROOT/results *(rw,sync,no_subtree_check,no_root_squash)

# Logs directory (read-write for workers)
$SHARED_ROOT/logs *(rw,sync,no_subtree_check,no_root_squash)

# Temporary workspace (read-write for workers)
$SHARED_ROOT/temp *(rw,sync,no_subtree_check,no_root_squash)

# Configuration templates (read-only for workers)
$SHARED_ROOT/configs *(ro,sync,no_subtree_check,no_root_squash)
EOF

# Apply NFS configuration
exportfs -ra
systemctl restart nfs-kernel-server

log_success "NFS exports configured and active"

# 4. Install management and monitoring tools
log_info "Installing management and monitoring tools..."

apt install -y \
    htop btop iotop \
    nethogs iftop \
    ncdu tree \
    rsync \
    logrotate \
    fail2ban \
    ufw \
    prometheus-node-exporter \
    nginx \
    certbot \
    python3-certbot-nginx

# Enable services
systemctl enable prometheus-node-exporter
systemctl enable nginx
systemctl enable fail2ban
systemctl start prometheus-node-exporter
systemctl start nginx
systemctl start fail2ban

log_success "Management tools installed"

# 5. Configure firewall for NFS and management
log_info "Configuring firewall..."

# Reset UFW to default
ufw --force reset

# Default policies
ufw default deny incoming
ufw default allow outgoing

# SSH access
ufw allow ssh

# NFS ports
ufw allow 2049/tcp    # NFS
ufw allow 111/tcp     # RPC portmapper
ufw allow 111/udp     # RPC portmapper
ufw allow 32765:32768/tcp  # NFS additional ports
ufw allow 32765:32768/udp  # NFS additional ports

# Hashmancer server ports
ufw allow 8080/tcp    # Main API
ufw allow 9090/tcp    # Metrics
ufw allow 80/tcp      # HTTP
ufw allow 443/tcp     # HTTPS

# Monitoring ports (restrict to local network if needed)
ufw allow 3000/tcp    # Grafana
ufw allow 9091/tcp    # Prometheus

# Database ports (restrict to local network)
ufw allow from 192.168.0.0/16 to any port 5432    # PostgreSQL
ufw allow from 10.0.0.0/8 to any port 5432        # PostgreSQL
ufw allow from 172.16.0.0/12 to any port 5432     # PostgreSQL

ufw --force enable

log_success "Firewall configured"

# 6. Create management scripts
log_info "Creating management scripts..."

mkdir -p /usr/local/bin/hashmancer

# Worker registration script
cat > /usr/local/bin/hashmancer/register-worker.sh << 'EOF'
#!/bin/bash
# Register a new worker node

WORKER_IP=$1
WORKER_NAME=$2

if [ -z "$WORKER_IP" ] || [ -z "$WORKER_NAME" ]; then
    echo "Usage: $0 <worker_ip> <worker_name>"
    exit 1
fi

echo "Registering worker: $WORKER_NAME ($WORKER_IP)"

# Add to known workers
echo "$WORKER_IP $WORKER_NAME" >> /srv/hashmancer/configs/known_workers.txt

# Create worker-specific directories
mkdir -p /srv/hashmancer/logs/workers/$WORKER_NAME
mkdir -p /srv/hashmancer/results/workers/$WORKER_NAME
chown -R hashmancer:hashmancer /srv/hashmancer/logs/workers/$WORKER_NAME
chown -R hashmancer:hashmancer /srv/hashmancer/results/workers/$WORKER_NAME

# Test NFS connectivity
echo "Testing NFS connectivity to $WORKER_IP..."
showmount -e $WORKER_IP 2>/dev/null && echo "✓ NFS client detected" || echo "⚠ No NFS client detected"

echo "Worker $WORKER_NAME registered successfully"
EOF

# System status script
cat > /usr/local/bin/hashmancer/status.sh << 'EOF'
#!/bin/bash
# Show comprehensive Hashmancer server status

echo "=============================================="
echo "Hashmancer Unified Server Status"
echo "=============================================="

echo
echo "📊 System Resources:"
echo "  CPU: $(nproc) cores, Load: $(uptime | awk -F'load average:' '{print $2}')"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $3"/"$2" ("$3/$2*100"%)"}' | bc -l | cut -d. -f1)%"
echo "  Disk: $(df -h /srv/hashmancer | tail -1 | awk '{print $3"/"$2" ("$5")"}')"

echo
echo "🗂️ NFS Status:"
systemctl is-active nfs-kernel-server >/dev/null && echo "  ✓ NFS Server: Running" || echo "  ✗ NFS Server: Stopped"
echo "  Active Exports: $(exportfs | wc -l)"
echo "  Connected Clients: $(netstat -an | grep :2049 | grep ESTABLISHED | wc -l)"

echo
echo "🖥️ Hashmancer Services:"
systemctl is-active hashmancer-server >/dev/null && echo "  ✓ Server: Running" || echo "  ⚠ Server: Not running"
systemctl is-active postgresql >/dev/null && echo "  ✓ Database: Running" || echo "  ✗ Database: Stopped"  
systemctl is-active redis >/dev/null && echo "  ✓ Redis: Running" || echo "  ✗ Redis: Stopped"

echo
echo "👥 Connected Workers:"
if [ -f /srv/hashmancer/configs/known_workers.txt ]; then
    cat /srv/hashmancer/configs/known_workers.txt | while read ip name; do
        ping -c1 -W1 $ip >/dev/null 2>&1 && status="✓ Online" || status="✗ Offline"
        echo "  $name ($ip): $status"
    done
else
    echo "  No workers registered yet"
fi

echo
echo "📈 Recent Activity:"
echo "  Active Jobs: $(find /srv/hashmancer/results/active -name "*.job" 2>/dev/null | wc -l)"
echo "  Completed Today: $(find /srv/hashmancer/results/completed -name "*.result" -mtime -1 2>/dev/null | wc -l)"
echo "  Log Size: $(du -sh /srv/hashmancer/logs 2>/dev/null | cut -f1)"

echo
echo "🔗 Service URLs:"
echo "  API: http://$(hostname -I | awk '{print $1}'):8080"
echo "  Monitoring: http://$(hostname -I | awk '{print $1}'):3000"
echo "  Metrics: http://$(hostname -I | awk '{print $1}'):9090"
EOF

# Cleanup script
cat > /usr/local/bin/hashmancer/cleanup.sh << 'EOF'
#!/bin/bash
# Cleanup old results and logs

echo "Starting Hashmancer cleanup..."

# Archive completed results older than 30 days
find /srv/hashmancer/results/completed -name "*.result" -mtime +30 -exec mv {} /srv/hashmancer/results/archived/ \;

# Compress old logs
find /srv/hashmancer/logs -name "*.log" -mtime +7 -exec gzip {} \;

# Remove old temporary files
find /srv/hashmancer/temp -mtime +1 -delete

# Remove failed jobs older than 7 days
find /srv/hashmancer/results/failed -name "*.job" -mtime +7 -delete

echo "Cleanup completed"
df -h /srv/hashmancer
EOF

# Make scripts executable
chmod +x /usr/local/bin/hashmancer/*.sh

# Create symlinks for easy access
ln -sf /usr/local/bin/hashmancer/status.sh /usr/local/bin/hashmancer-status
ln -sf /usr/local/bin/hashmancer/register-worker.sh /usr/local/bin/hashmancer-register
ln -sf /usr/local/bin/hashmancer/cleanup.sh /usr/local/bin/hashmancer-cleanup

log_success "Management scripts created"

# 7. Configure log rotation
log_info "Configuring log rotation..."

cat > /etc/logrotate.d/hashmancer << 'EOF'
/srv/hashmancer/logs/server/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su hashmancer hashmancer
}

/srv/hashmancer/logs/workers/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su hashmancer hashmancer
}

/srv/hashmancer/logs/system/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su hashmancer hashmancer
}
EOF

log_success "Log rotation configured"

# 8. Create systemd service for Hashmancer server
log_info "Creating Hashmancer server service..."

cat > /etc/systemd/system/hashmancer-server.service << EOF
[Unit]
Description=Hashmancer Coordination Server
After=network.target postgresql.service redis.service nfs-kernel-server.service
Wants=postgresql.service redis.service nfs-kernel-server.service

[Service]
Type=simple
User=$HASHMANCER_USER
Group=$HASHMANCER_USER
WorkingDirectory=$HASHMANCER_HOME/hashmancer/server
Environment=PYTHONPATH=$HASHMANCER_HOME/hashmancer
Environment=HASHMANCER_CONFIG=/srv/hashmancer/configs/server/config.yaml
Environment=HASHMANCER_SHARED_ROOT=/srv/hashmancer
ExecStart=$HASHMANCER_HOME/.venv/hashmancer/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8080
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable hashmancer-server

log_success "Hashmancer server service configured"

# 9. Create monitoring configuration
log_info "Setting up monitoring configuration..."

mkdir -p /srv/hashmancer/configs/{prometheus,grafana}

# Prometheus configuration
cat > /srv/hashmancer/configs/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'hashmancer-server'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s
    metrics_path: '/metrics'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 10s

  - job_name: 'hashmancer-workers'
    file_sd_configs:
      - files: ['/srv/hashmancer/configs/prometheus/workers.json']
        refresh_interval: 30s

rule_files:
  - "/srv/hashmancer/configs/prometheus/alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
EOF

# Worker discovery template
cat > /srv/hashmancer/configs/prometheus/workers.json << 'EOF'
[
  {
    "targets": [],
    "labels": {
      "job": "hashmancer-worker"
    }
  }
]
EOF

# Basic alerts
cat > /srv/hashmancer/configs/prometheus/alerts.yml << 'EOF'
groups:
  - name: hashmancer-alerts
    rules:
      - alert: WorkerDown
        expr: up{job="hashmancer-worker"} == 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Hashmancer worker is down"
          description: "Worker {{ $labels.instance }} has been down for more than 1 minute."

      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes."

      - alert: LowDiskSpace
        expr: node_filesystem_avail_bytes{mountpoint="/srv/hashmancer"} / node_filesystem_size_bytes{mountpoint="/srv/hashmancer"} * 100 < 10
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Less than 10% disk space remaining on Hashmancer storage."
EOF

chown -R $HASHMANCER_USER:$HASHMANCER_USER /srv/hashmancer/configs

log_success "Monitoring configuration created"

# 10. Create sample datasets
log_info "Creating sample datasets..."

su - $HASHMANCER_USER << 'EOF'
# Create sample wordlists
cat > /srv/hashmancer/datasets/wordlists/common_passwords.txt << 'WORDLIST'
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
WORDLIST

# Create Best64 rules
cat > /srv/hashmancer/datasets/rules/best64.rule << 'RULES'
:
l
u
c
C
t
r
d
f
{
}
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
so0
si1
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
RULES

# Create sample test hashes
cat > /srv/hashmancer/datasets/hashes/test_md5.txt << 'HASHES'
5d41402abc4b2a76b9719d911017c592:hello
098f6bcd4621d373cade4e832627b4f6:test  
e99a18c428cb38d5f260853678922e03:abc123
5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8:password
HASHES

EOF

log_success "Sample datasets created"

# 11. Create worker connection helper
log_info "Creating worker connection helper..."

cat > $HASHMANCER_HOME/mount-hashmancer.sh << EOF
#!/bin/bash
# Helper script for workers to mount Hashmancer NFS shares

SERVER_IP=\${1:-$(hostname -I | awk '{print $1}')}

if [ "\$EUID" -ne 0 ]; then
    echo "This script needs to be run as root on the worker node"
    exit 1
fi

echo "Mounting Hashmancer NFS shares from \$SERVER_IP..."

# Install NFS client if not present
apt update && apt install -y nfs-common

# Create mount points
mkdir -p /mnt/hashmancer/{datasets,results,logs,configs,temp}

# Mount shares
mount -t nfs \$SERVER_IP:/srv/hashmancer/datasets /mnt/hashmancer/datasets
mount -t nfs \$SERVER_IP:/srv/hashmancer/results /mnt/hashmancer/results
mount -t nfs \$SERVER_IP:/srv/hashmancer/logs /mnt/hashmancer/logs
mount -t nfs \$SERVER_IP:/srv/hashmancer/configs /mnt/hashmancer/configs
mount -t nfs \$SERVER_IP:/srv/hashmancer/temp /mnt/hashmancer/temp

# Add to fstab for persistent mounting
echo "\$SERVER_IP:/srv/hashmancer/datasets /mnt/hashmancer/datasets nfs defaults 0 0" >> /etc/fstab
echo "\$SERVER_IP:/srv/hashmancer/results /mnt/hashmancer/results nfs defaults 0 0" >> /etc/fstab
echo "\$SERVER_IP:/srv/hashmancer/logs /mnt/hashmancer/logs nfs defaults 0 0" >> /etc/fstab
echo "\$SERVER_IP:/srv/hashmancer/configs /mnt/hashmancer/configs nfs defaults 0 0" >> /etc/fstab
echo "\$SERVER_IP:/srv/hashmancer/temp /mnt/hashmancer/temp nfs defaults 0 0" >> /etc/fstab

echo "✓ NFS shares mounted successfully"
echo "  Datasets: /mnt/hashmancer/datasets (read-only)"
echo "  Results: /mnt/hashmancer/results (read-write)"
echo "  Logs: /mnt/hashmancer/logs (read-write)"
echo "  Configs: /mnt/hashmancer/configs (read-only)"
echo "  Temp: /mnt/hashmancer/temp (read-write)"
EOF

chown $HASHMANCER_USER:$HASHMANCER_USER $HASHMANCER_HOME/mount-hashmancer.sh
chmod +x $HASHMANCER_HOME/mount-hashmancer.sh

log_success "Worker connection helper created"

# 12. Final system configuration
log_info "Applying final system configuration..."

# Increase NFS server thread count for better performance
echo 'RPCNFSDCOUNT=16' >> /etc/default/nfs-kernel-server

# Optimize NFS performance
cat >> /etc/sysctl.conf << 'EOF'

# NFS Performance Optimizations
net.core.rmem_default = 262144
net.core.rmem_max = 16777216
net.core.wmem_default = 262144
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 65536 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
EOF

sysctl -p

# Restart services to apply changes
systemctl restart nfs-kernel-server
systemctl restart rpcbind

log_success "System configuration applied"

# Create a summary
echo
echo "=============================================="
log_success "Hashmancer Unified Server Setup Complete!"
echo "=============================================="
echo
echo "🏗️ Services Configured:"
echo "   ✓ NFS Server - Shared storage for workers"
echo "   ✓ Management Tools - Status, monitoring, cleanup"
echo "   ✓ Firewall - Secure access configured"  
echo "   ✓ Log Rotation - Automatic log management"
echo "   ✓ Systemd Service - Auto-start Hashmancer server"
echo
echo "📁 Shared Storage Structure:"
echo "   /srv/hashmancer/datasets/   - Wordlists, rules, hashes"
echo "   /srv/hashmancer/results/    - Job results and archives"
echo "   /srv/hashmancer/logs/       - System and worker logs"
echo "   /srv/hashmancer/configs/    - Configuration templates"
echo "   /srv/hashmancer/temp/       - Temporary workspace"
echo
echo "🛠️ Management Commands:"
echo "   hashmancer-status           - Show system status"
echo "   hashmancer-register <ip> <name> - Register new worker"  
echo "   hashmancer-cleanup          - Cleanup old files"
echo
echo "🔗 For Workers:"
echo "   Copy and run: $HASHMANCER_HOME/mount-hashmancer.sh"
echo
echo "📊 Monitoring:"
echo "   NFS exports: exportfs -v"
echo "   Connected clients: showmount -a"
echo "   System status: hashmancer-status"
echo
echo "🚀 Next Steps:"
echo "   1. Start your Hashmancer application development"
echo "   2. Register worker nodes with: hashmancer-register <worker_ip> <worker_name>"
echo "   3. Monitor via: http://$(hostname -I | awk '{print $1}'):3000"
echo
log_success "Ready for Hashmancer deployment! 🎉"