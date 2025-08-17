# Ubiquiti Dream Router VPN Setup for Vast.ai Workers

Complete guide to connect Vast.ai workers to your local Hashmancer server through UDR VPN.

## Overview

**Goal:** Allow Vast.ai GPU workers to securely connect to your local Hashmancer server running behind Ubiquiti Dream Router.

**Architecture:**
```
[Vast.ai Worker] --VPN--> [UDR VPN Server] --LAN--> [Local Hashmancer Server]
```

## VPN Options for UDR

### Option 1: WireGuard (Recommended)
- **Pros:** Modern, fast, built into UDR, excellent for containers
- **Cons:** Requires UniFi OS 3.0+
- **Best for:** Docker containers, high performance

### Option 2: OpenVPN
- **Pros:** Widely supported, works with older systems
- **Cons:** Slower than WireGuard, more complex setup
- **Best for:** Legacy compatibility

### Option 3: L2TP/IPSec
- **Pros:** Built-in client support in most systems
- **Cons:** Less secure, more complex NAT traversal
- **Best for:** Simple client setup

**Recommendation:** Use WireGuard for Vast.ai workers due to performance and Docker compatibility.

---

## Method 1: WireGuard Setup (Recommended)

### Step 1: Enable WireGuard on UDR

1. **Access UniFi Network Console**
   ```
   https://your-udr-ip
   Login with admin credentials
   ```

2. **Enable WireGuard VPN**
   - Go to **Settings** → **VPN**
   - Select **VPN Server**
   - Choose **WireGuard**
   - Click **Create VPN Server**

3. **Configure WireGuard Server**
   ```yaml
   Name: hashmancer-vpn
   Port: 51820 (default)
   Network: 10.8.0.0/24
   DNS Server: 192.168.1.1 (your UDR IP)
   ```

4. **Create Client Profiles**
   - Click **Add Client**
   - Name: `vast-worker-1`
   - Generate keys automatically
   - Download `.conf` file

### Step 2: Configure UDR Firewall

1. **Create Firewall Rules**
   - Go to **Settings** → **Security** → **Firewall Rules**
   - Create **WAN Local** rule:
     ```yaml
     Name: Allow WireGuard
     Rule Applied: Before predefined rules
     Type: Internet In
     Protocol: UDP
     Port: 51820
     Action: Accept
     ```

2. **Create VPN to LAN Access Rule**
   - Create **LAN Local** rule:
     ```yaml
     Name: VPN to Hashmancer
     Source: VPN Network (10.8.0.0/24)
     Destination: LAN
     Port: 8080,6379
     Action: Accept
     ```

### Step 3: Port Forwarding (Optional)
If you need direct access without VPN routing:
- Go to **Settings** → **Advanced Features** → **Port Forwarding**
- Create rule for Hashmancer server:
  ```yaml
  Name: Hashmancer Server
  From: WAN
  Port: 8080
  Forward IP: 192.168.1.XXX (your server IP)
  Forward Port: 8080
  Protocol: TCP
  ```

---

## Method 2: OpenVPN Setup (Alternative)

### Step 1: Enable OpenVPN on UDR

1. **Create OpenVPN Server**
   - Go to **Settings** → **VPN** → **VPN Server**
   - Select **OpenVPN**
   - Configure:
     ```yaml
     Protocol: UDP
     Port: 1194
     Encryption: AES-256
     Network: 192.168.100.0/24
     ```

2. **Download Client Config**
   - Generate client certificate
   - Download `.ovpn` file

### Step 2: Configure OpenVPN Client on Vast.ai

Create `openvpn-client.conf` from downloaded file and modify:
```conf
# Add these lines to handle Docker networking
route-nopull
route 192.168.1.0 255.255.255.0
dhcp-option DNS 192.168.1.1
```

---

## Vast.ai Worker VPN Configuration

### WireGuard Docker Container

Create `Dockerfile.worker-vpn`:
```dockerfile
FROM yourusername/hashmancer-worker:latest

# Install WireGuard
RUN apt-get update && apt-get install -y \
    wireguard-tools \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Copy WireGuard config
COPY wg0.conf /etc/wireguard/wg0.conf

# Create startup script
RUN echo '#!/bin/bash\n\
# Start WireGuard\n\
wg-quick up wg0\n\
# Start Hashmancer worker\n\
python -m hashmancer.worker.hashmancer_worker.worker_agent\n\
' > /start-vpn-worker.sh && chmod +x /start-vpn-worker.sh

CMD ["/start-vpn-worker.sh"]
```

### WireGuard Client Config (`wg0.conf`)
```ini
[Interface]
PrivateKey = YOUR_PRIVATE_KEY_FROM_UDR
Address = 10.8.0.2/32
DNS = 192.168.1.1

[Peer]
PublicKey = YOUR_UDR_PUBLIC_KEY
Endpoint = YOUR_PUBLIC_IP:51820
AllowedIPs = 192.168.1.0/24
PersistentKeepalive = 25
```

### Build and Deploy VPN Worker
```bash
# Build VPN-enabled worker
docker build -f Dockerfile.worker-vpn -t yourusername/hashmancer-worker-vpn:latest .

# Push to Docker Hub
docker push yourusername/hashmancer-worker-vpn:latest

# Deploy on Vast.ai with privileges
docker run -d \
  --cap-add=NET_ADMIN \
  --cap-add=SYS_MODULE \
  -e SERVER_URL=http://192.168.1.XXX:8080 \
  -e REDIS_URL=redis://192.168.1.XXX:6379 \
  yourusername/hashmancer-worker-vpn:latest
```

---

## Alternative: SSH Tunnel Method (Simpler)

If VPN seems complex, use SSH tunneling:

### Step 1: Enable SSH on UDR
1. **Enable SSH Access**
   - Go to **Settings** → **System** → **Console**
   - Enable **SSH**
   - Set strong password

### Step 2: Create SSH Tunnel Docker

Create `Dockerfile.worker-ssh`:
```dockerfile
FROM yourusername/hashmancer-worker:latest

# Install SSH client
RUN apt-get update && apt-get install -y \
    openssh-client \
    autossh \
    && rm -rf /var/lib/apt/lists/*

# Create SSH tunnel script
RUN echo '#!/bin/bash\n\
# Create SSH tunnel to home network\n\
autossh -M 0 -N -f -L 8080:192.168.1.XXX:8080 -L 6379:192.168.1.XXX:6379 admin@YOUR_PUBLIC_IP\n\
# Start worker connecting to localhost (tunneled)\n\
python -m hashmancer.worker.hashmancer_worker.worker_agent\n\
' > /start-ssh-worker.sh && chmod +x /start-ssh-worker.sh

CMD ["/start-ssh-worker.sh"]
```

### Step 3: Deploy SSH Tunnel Worker
```bash
# Build SSH worker
docker build -f Dockerfile.worker-ssh -t yourusername/hashmancer-worker-ssh:latest .

# Deploy with SSH key
docker run -d \
  -v ~/.ssh:/root/.ssh:ro \
  -e SERVER_URL=http://localhost:8080 \
  -e REDIS_URL=redis://localhost:6379 \
  yourusername/hashmancer-worker-ssh:latest
```

---

## Dynamic DNS Setup (Recommended)

Since your home IP might change, set up dynamic DNS:

### Using UniFi Dynamic DNS
1. **Configure in UDR**
   - Go to **Settings** → **Internet** → **Dynamic DNS**
   - Choose provider (Cloudflare, No-IP, DuckDNS)
   - Set hostname: `hashmancer.yourdomain.com`

### Using Cloudflare (Free)
```bash
# Install cloudflare-ddns on a local machine
pip install cloudflare-ddns

# Configure
cloudflare-ddns --email your@email.com --api-key YOUR_API_KEY --domain hashmancer.yourdomain.com
```

---

## Security Considerations

### Firewall Rules
```yaml
# Only allow VPN traffic to specific ports
WAN In Rules:
- Allow UDP 51820 (WireGuard)
- Allow TCP 22 (SSH - if using SSH method)
- Block all other WAN to LAN

VPN to LAN Rules:
- Allow VPN to 192.168.1.XXX:8080 (Hashmancer)
- Allow VPN to 192.168.1.XXX:6379 (Redis)
- Block VPN to other LAN devices
```

### Authentication
```bash
# Use strong pre-shared keys for WireGuard
wg genkey | tee privatekey | wg pubkey > publickey

# For SSH method, use key-based auth only
ssh-keygen -t ed25519 -f vast_worker_key
```

---

## Vast.ai Deployment Scripts

### WireGuard Vast.ai Template
```bash
#!/bin/bash
# Vast.ai startup script for VPN workers

# Install Docker if needed
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Download WireGuard config from your server
curl -o wg0.conf https://yourdomain.com/vpn-configs/wg0.conf

# Run VPN worker
docker run -d \
  --cap-add=NET_ADMIN \
  --cap-add=SYS_MODULE \
  --name hashmancer-worker-vpn \
  -v $(pwd)/wg0.conf:/etc/wireguard/wg0.conf \
  -e SERVER_URL=http://192.168.1.XXX:8080 \
  -e REDIS_URL=redis://192.168.1.XXX:6379 \
  yourusername/hashmancer-worker-vpn:latest

# Monitor worker
docker logs -f hashmancer-worker-vpn
```

### Auto-Scaling with VPN
```bash
#!/bin/bash
# Scale VPN workers based on queue

SERVER_URL="http://192.168.1.XXX:8080"
DOCKER_IMAGE="yourusername/hashmancer-worker-vpn:latest"

create_vpn_worker() {
    vastai create instance \
        --image nvidia/cuda:11.8-devel-ubuntu20.04 \
        --onstart-cmd "curl -s https://yourdomain.com/scripts/start-vpn-worker.sh | bash"
}

# Monitor and scale
while true; do
    QUEUE_SIZE=$(curl -s ${SERVER_URL}/api/queue/size)
    if [ "$QUEUE_SIZE" -gt 5 ]; then
        create_vpn_worker
    fi
    sleep 60
done
```

---

## Troubleshooting

### Common Issues

#### 1. VPN Connection Fails
```bash
# Check UDR logs
ssh admin@your-udr-ip
tail -f /var/log/messages | grep wireguard

# Test connectivity from worker
docker exec worker ping 192.168.1.1
```

#### 2. Can't Reach Hashmancer Server
```bash
# Check routes in worker container
docker exec worker ip route show
docker exec worker ping 192.168.1.XXX

# Check UDR firewall logs
# UniFi Console → Insights → Events → Firewall
```

#### 3. DNS Resolution Issues
```bash
# Test DNS in worker
docker exec worker nslookup google.com 192.168.1.1
docker exec worker cat /etc/resolv.conf
```

### Testing Connectivity

#### From Vast.ai Worker
```bash
# Test VPN tunnel
docker exec worker ping 192.168.1.1

# Test Hashmancer server
docker exec worker curl http://192.168.1.XXX:8080/health

# Test Redis
docker exec worker redis-cli -h 192.168.1.XXX ping
```

#### Monitor VPN Status
```bash
# WireGuard status on UDR
ssh admin@your-udr-ip
wg show

# Check active connections
netstat -an | grep 51820
```

---

## Performance Optimization

### WireGuard Tuning
```ini
# In wg0.conf, add for better performance
[Interface]
MTU = 1420
PostUp = echo 'net.core.default_qdisc=fq' >> /etc/sysctl.conf; echo 'net.ipv4.tcp_congestion_control=bbr' >> /etc/sysctl.conf; sysctl -p
```

### Network Optimization
```bash
# On Vast.ai worker containers
docker run --sysctl net.core.rmem_max=26214400 \
  --sysctl net.core.rmem_default=26214400 \
  --sysctl net.core.wmem_max=26214400 \
  --sysctl net.core.wmem_default=26214400 \
  yourusername/hashmancer-worker-vpn:latest
```

---

## Quick Setup Summary

**Recommended Path: WireGuard**

1. **UDR Setup:**
   - Enable WireGuard VPN server
   - Create client configs
   - Configure firewall rules

2. **Build VPN Worker:**
   ```bash
   docker build -f Dockerfile.worker-vpn -t yourusername/hashmancer-worker-vpn .
   docker push yourusername/hashmancer-worker-vpn:latest
   ```

3. **Deploy on Vast.ai:**
   ```bash
   docker run -d --cap-add=NET_ADMIN --cap-add=SYS_MODULE \
     -e SERVER_URL=http://192.168.1.XXX:8080 \
     yourusername/hashmancer-worker-vpn:latest
   ```

This setup provides secure, encrypted access to your local Hashmancer server from any Vast.ai GPU worker worldwide.