# 🌊 DigitalOcean Hashmancer Server Setup

This guide shows you how to set up your DigitalOcean droplet as a Hashmancer server for Vast.ai workers to connect to.

## 🚀 **Quick Setup (5 Minutes)**

### **Step 1: Create DigitalOcean Droplet**

**Recommended Specs:**
- **Size**: Basic ($12/month) or CPU-Optimized ($18/month)
- **Memory**: 2GB minimum, 4GB recommended
- **Storage**: 50GB minimum
- **OS**: Ubuntu 22.04 LTS
- **Region**: Choose closest to your Vast.ai workers

**In DigitalOcean Console:**
1. Click **"Create Droplet"**
2. **Image**: Ubuntu 22.04 LTS
3. **Size**: Basic $12/month (2GB RAM, 1 CPU, 50GB SSD)
4. **Region**: Choose your preferred region
5. **Authentication**: SSH Key (recommended) or Password
6. **Hostname**: `hashmancer-server`
7. Click **"Create Droplet"**

### **Step 2: Connect to Your Droplet**

```bash
# SSH to your droplet (replace with your IP)
ssh root@your-droplet-ip
```

### **Step 3: Upload and Run Setup Script**

```bash
# Upload the setup script
curl -o setup-digitalocean-server.sh https://raw.githubusercontent.com/yourusername/hashmancer/main/scripts/setup-digitalocean-server.sh

# Or copy from your local machine
scp scripts/setup-digitalocean-server.sh root@your-droplet-ip:/tmp/

# Make executable and run
chmod +x setup-digitalocean-server.sh
./setup-digitalocean-server.sh
```

### **Step 4: Verify Setup**

After the script completes, you'll see:
```
🎉 Hashmancer server setup complete!
Your server is ready for Vast.ai workers at: http://YOUR_IP:8080

🌐 Server Access Information:
   API Endpoint: http://YOUR_IP:8080
   Health Check: http://YOUR_IP:8080/health
   
🎯 For Vast.ai Workers:
   Set HASHMANCER_SERVER_IP=YOUR_IP
```

Test your server:
```bash
# Test health endpoint
curl http://YOUR_IP:8080/health

# Should return: {"status":"healthy",...}
```

## 🛠️ **Manual Setup (Alternative)**

If you prefer manual setup or need to customize:

### **1. Prepare System**
```bash
# Update system
apt-get update && apt-get upgrade -y

# Install dependencies
apt-get install -y python3 python3-pip python3-venv nginx redis-server ufw curl wget git
```

### **2. Configure Firewall**
```bash
# Configure UFW firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 8080
ufw allow 80
ufw allow 443
ufw --force enable
```

### **3. Create Hashmancer User**
```bash
# Create user and directories
useradd -m -s /bin/bash hashmancer
mkdir -p /opt/hashmancer
chown hashmancer:hashmancer /opt/hashmancer
```

### **4. Setup Python Environment**
```bash
# Switch to hashmancer user
sudo -u hashmancer bash

# Create virtual environment
cd /opt/hashmancer
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install fastapi uvicorn redis requests click tabulate psutil aiohttp
```

### **5. Copy Hashmancer Files**
```bash
# Copy your hashmancer directory to /opt/hashmancer/
scp -r hashmancer/ root@your-droplet-ip:/opt/hashmancer/
chown -R hashmancer:hashmancer /opt/hashmancer
```

### **6. Create Systemd Service**
```bash
cat > /etc/systemd/system/hashmancer.service << 'EOF'
[Unit]
Description=Hashmancer Server
After=network.target redis.service

[Service]
Type=simple
User=hashmancer
Group=hashmancer
WorkingDirectory=/opt/hashmancer
Environment=PATH=/opt/hashmancer/venv/bin
ExecStart=/opt/hashmancer/venv/bin/python -m hashmancer.server.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable hashmancer
systemctl start hashmancer
```

## 📊 **Server Management**

Use the management script for easy control:

```bash
# Make management script available
chmod +x scripts/manage-server.sh
cp scripts/manage-server.sh /usr/local/bin/hashmancer-manage

# Common commands
sudo hashmancer-manage status    # Check server status
sudo hashmancer-manage restart   # Restart server
sudo hashmancer-manage logs      # View logs
sudo hashmancer-manage workers   # Show connected workers
sudo hashmancer-manage health    # Health check
sudo hashmancer-manage monitor   # Real-time monitoring
```

## 🔧 **Configuration**

### **Environment Variables**
Create `/etc/hashmancer/config.env`:
```bash
HASHMANCER_HOST=0.0.0.0
HASHMANCER_PORT=8080
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
MAX_WORKERS=100
```

### **Redis Configuration**
Edit `/etc/redis/redis.conf`:
```bash
# Optimize for Hashmancer
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### **Nginx Reverse Proxy** (Optional)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## 🎯 **Connecting Vast.ai Workers**

Once your server is running, use this in your Vast.ai templates:

### **Environment Variables:**
```
HASHMANCER_SERVER_IP=your-droplet-ip
HASHMANCER_SERVER_PORT=8080
```

### **Docker Image:**
```
yourusername/hashmancer-worker:latest
```

### **Test Connection:**
```bash
# From Vast.ai worker
curl http://your-droplet-ip:8080/health

# Register worker
curl -X POST http://your-droplet-ip:8080/worker/register \
  -H "Content-Type: application/json" \
  -d '{"worker_id":"test-worker","status":"ready"}'
```

## 📈 **Monitoring & Maintenance**

### **Check Server Status**
```bash
# Service status
systemctl status hashmancer

# Resource usage
htop
df -h
free -h

# Network connections
netstat -tlnp | grep :8080
```

### **View Logs**
```bash
# Service logs
journalctl -u hashmancer -f

# Nginx logs
tail -f /var/log/nginx/access.log

# System logs
tail -f /var/log/syslog
```

### **Connected Workers**
```bash
# Via API
curl http://localhost:8080/workers | python3 -m json.tool

# Via management script
sudo hashmancer-manage workers
```

### **Backup Configuration**
```bash
# Create backup
sudo hashmancer-manage backup

# Backups stored in /var/backups/hashmancer/
```

## 🔒 **Security Best Practices**

### **Firewall Rules**
```bash
# Only allow necessary ports
ufw status numbered
ufw delete [rule-number]  # Remove unnecessary rules
```

### **SSL/TLS** (Production)
```bash
# Install Certbot
snap install --classic certbot

# Get certificate
certbot --nginx -d your-domain.com
```

### **Regular Updates**
```bash
# Update system
apt-get update && apt-get upgrade -y

# Update Hashmancer
sudo hashmancer-manage update
```

### **Monitor Failed Login Attempts**
```bash
# Install fail2ban
apt-get install fail2ban

# Configure for SSH protection
systemctl enable fail2ban
```

## 💰 **Cost Optimization**

### **DigitalOcean Costs:**
- **Basic**: $12/month (suitable for moderate use)
- **CPU-Optimized**: $18/month (better for high worker loads)
- **Memory-Optimized**: $24/month (large job queues)

### **Monitoring Costs:**
```bash
# Check current usage
doctl compute droplet list
doctl monitoring bandwidth
```

### **Auto-scaling** (Advanced)
```bash
# Scale workers based on queue size
if [ $(curl -s localhost:8080/jobs | jq '. | length') -gt 10 ]; then
    # Deploy more Vast.ai workers
    ./scripts/vast-deploy-prebuilt.sh --count 2
fi
```

## 🚨 **Troubleshooting**

### **Server Won't Start**
```bash
# Check logs
journalctl -u hashmancer -n 50

# Check port conflicts
sudo lsof -i :8080

# Check permissions
ls -la /opt/hashmancer/
```

### **Workers Can't Connect**
```bash
# Test from external network
curl http://your-droplet-ip:8080/health

# Check firewall
ufw status
iptables -L

# Check if port is open
nmap -p 8080 your-droplet-ip
```

### **High Resource Usage**
```bash
# Check processes
htop
ps aux | grep hashmancer

# Check disk space
df -h
du -sh /opt/hashmancer/*

# Check memory
free -h
cat /proc/meminfo
```

## 🎉 **Success Indicators**

Your server is working correctly when:

✅ **Health check returns HTTP 200**
✅ **Workers appear in `/workers` endpoint**
✅ **Server logs show worker registrations**
✅ **System resources are stable**
✅ **External connectivity works**

**Test command:**
```bash
curl -s http://your-droplet-ip:8080/health | jq .
```

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "Hashmancer Server",
  "workers": 3,
  "jobs": 5
}
```

Your DigitalOcean droplet is now ready to manage Vast.ai workers! 🌊🚀