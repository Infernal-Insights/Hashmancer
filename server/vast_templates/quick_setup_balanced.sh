#!/bin/bash
# Quick Setup Script - Balanced Performance
set -e

echo "🚀 Hashmancer Balanced Setup Starting..."

# Download optimized wordlists for balanced performance
echo "📚 Setting up wordlists..."
cd /workspace/hashmancer/wordlists

# Download RockYou if not present
if [ ! -f "rockyou.txt" ]; then
    echo "Downloading RockYou wordlist..."
    wget -q "https://github.com/danielmiessler/SecLists/raw/master/Passwords/Leaked-Databases/rockyou.txt.tar.gz"
    tar -xzf rockyou.txt.tar.gz
    rm rockyou.txt.tar.gz
fi

# Download common passwords
if [ ! -f "common-10k.txt" ]; then
    echo "Downloading common passwords..."
    wget -q -O common-10k.txt "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/10-million-password-list-top-10000.txt"
fi

# Download medium wordlist
if [ ! -f "weakpass-2a.txt" ]; then
    echo "Downloading WeakPass wordlist..."
    wget -q "https://download.weakpass.com/wordlists/1949/weakpass_2a.txt.gz"
    gunzip weakpass_2a.txt.gz
fi

# Set up rules
echo "⚙️ Setting up hashcat rules..."
cd /workspace/hashmancer/rules

# Copy built-in rules if available
if [ -d "/usr/share/hashcat/rules" ]; then
    cp /usr/share/hashcat/rules/*.rule . 2>/dev/null || true
fi

# Download additional rules
if [ ! -f "dive.rule" ]; then
    wget -q -O dive.rule "https://raw.githubusercontent.com/hashcat/hashcat/master/rules/dive.rule"
fi

if [ ! -f "leetspeak.rule" ]; then
    wget -q -O leetspeak.rule "https://raw.githubusercontent.com/hashcat/hashcat/master/rules/leetspeak.rule"
fi

# Optimize GPU settings for balanced performance
echo "🔧 Optimizing GPU settings..."
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -pl 250  # Set power limit to 250W for RTX cards

# Set up hashcat optimizations
echo "Creating hashcat optimization config..."
mkdir -p ~/.hashcat
cat > ~/.hashcat/hashcat.hcstat2 << 'EOF'
# Hashcat markov chains for better performance
# This will be populated by hashcat on first run
EOF

# Create performance monitoring script
cat > /workspace/hashmancer/monitor_performance.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
    echo "Memory Usage:"
    free -h | grep Mem
    echo "Load Average:"
    uptime
    echo ""
    sleep 30
done
EOF
chmod +x /workspace/hashmancer/monitor_performance.sh

# Set optimal environment variables
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB cache
export HASHCAT_FORCE=1

echo "✅ Balanced setup complete!"
echo "📊 System ready for hash cracking with optimized performance"
echo "💡 Available wordlists: rockyou.txt, common-10k.txt, weakpass-2a.txt"
echo "🎯 GPU optimization: Enabled"
echo "📈 Performance monitoring: Available via ./monitor_performance.sh"