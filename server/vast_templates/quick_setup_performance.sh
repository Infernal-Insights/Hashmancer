#!/bin/bash
# Quick Setup Script - High Performance
set -e

echo "🚀 Hashmancer High Performance Setup Starting..."

# Download comprehensive wordlists for maximum coverage
echo "📚 Setting up comprehensive wordlists..."
cd /workspace/hashmancer/wordlists

# Download all major wordlists in parallel
download_wordlist() {
    local name=$1
    local url=$2
    local extract_cmd=$3
    
    if [ ! -f "$name" ]; then
        echo "Downloading $name..."
        wget -q "$url" -O "${name}.tmp"
        if [ -n "$extract_cmd" ]; then
            $extract_cmd "${name}.tmp"
            rm "${name}.tmp"
        else
            mv "${name}.tmp" "$name"
        fi
        echo "✅ $name ready"
    fi
}

# Download wordlists in parallel for speed
{
    download_wordlist "rockyou.txt" "https://github.com/danielmiessler/SecLists/raw/master/Passwords/Leaked-Databases/rockyou.txt.tar.gz" "tar -xzf"
} &

{
    download_wordlist "crackstation.txt" "https://crackstation.net/files/crackstation.txt.gz" "gunzip"
} &

{
    download_wordlist "weakpass_2a.txt" "https://download.weakpass.com/wordlists/1949/weakpass_2a.txt.gz" "gunzip"
} &

{
    download_wordlist "kaonashi.txt" "https://download.weakpass.com/wordlists/1948/kaonashi.txt.gz" "gunzip"
} &

# Wait for downloads to complete
wait

echo "📊 Wordlist summary:"
echo "Total wordlist size: $(du -sh *.txt | awk '{sum+=$1} END {print sum "MB"}')"
echo "Files available: $(ls -1 *.txt | wc -l) wordlists"

# Set up comprehensive rules
echo "⚙️ Setting up comprehensive hashcat rules..."
cd /workspace/hashmancer/rules

# Download all major rule sets
rule_urls=(
    "https://raw.githubusercontent.com/hashcat/hashcat/master/rules/best64.rule"
    "https://raw.githubusercontent.com/hashcat/hashcat/master/rules/dive.rule"
    "https://raw.githubusercontent.com/hashcat/hashcat/master/rules/leetspeak.rule"
    "https://raw.githubusercontent.com/NotSoSecure/password_cracking_rules/master/OneRuleToRuleThemAll.rule"
    "https://raw.githubusercontent.com/stealthsploit/OneRuleToRuleThemStill/main/OneRuleToRuleThemStill.rule"
)

for url in "${rule_urls[@]}"; do
    filename=$(basename "$url")
    if [ ! -f "$filename" ]; then
        wget -q "$url" -O "$filename" &
    fi
done
wait

# Create custom high-performance rule
cat > high_performance.rule << 'EOF'
# High Performance Custom Rules
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
$0
$1
$2
$3
$4
$5
$6
$7
$8
$9
$!
$@
$#
$$
$%
^1
^2
^3
^a
^A
EOF

# Maximize GPU performance
echo "🔧 Maximizing GPU performance..."

# Enable persistence mode and set maximum power
nvidia-smi -pm 1
nvidia-smi -pl $(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | head -1)

# Set maximum memory and compute clocks
nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits | head -1 | tr ',' ' ')

# Optimize system for maximum performance
echo "🔧 System optimizations..."

# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase file limits
ulimit -n 65536

# Set optimal environment variables for performance
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=4294967296  # 4GB cache
export HASHCAT_FORCE=1
export HASHCAT_WORKLOAD_PROFILE=4  # Nightmare mode
export HASHCAT_KERNEL_ACCEL=1024
export HASHCAT_KERNEL_LOOPS=1024

# Create advanced performance monitoring
cat > /workspace/hashmancer/advanced_monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== HASHMANCER PERFORMANCE MONITOR ==="
    echo "Time: $(date)"
    echo ""
    
    echo "🎯 GPU Performance:"
    nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory --format=csv,noheader
    echo ""
    
    echo "💾 Memory Status:"
    free -h
    echo ""
    
    echo "🔥 CPU Status:"
    echo "Load: $(uptime | awk -F'load average:' '{print $2}')"
    echo "Temp: $(sensors 2>/dev/null | grep 'Core 0' | awk '{print $3}' || echo 'N/A')"
    echo ""
    
    echo "📊 Process Status:"
    ps aux | grep -E "(hashcat|darkling)" | grep -v grep | head -5
    echo ""
    
    echo "🌡️ Thermal Status:"
    nvidia-smi --query-gpu=temperature.gpu,temperature.memory --format=csv,noheader,nounits | while read temp; do
        echo "GPU: ${temp}°C"
    done
    
    sleep 5
done
EOF
chmod +x /workspace/hashmancer/advanced_monitor.sh

# Create benchmark script
cat > /workspace/hashmancer/benchmark.sh << 'EOF'
#!/bin/bash
echo "🚀 Running Hashmancer Performance Benchmark..."

# Create test hash file
echo "5d41402abc4b2a76b9719d911017c592" > test_hash.txt  # MD5 of "hello"

echo "📊 Benchmark Results:"
echo "==================="

# Test different hash types
echo "MD5 Performance:"
timeout 30s hashcat -m 0 test_hash.txt rockyou.txt --quiet --potfile-disable 2>/dev/null | tail -1

echo "SHA1 Performance:"
echo "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d" > test_sha1.txt  # SHA1 of "hello"
timeout 30s hashcat -m 100 test_sha1.txt rockyou.txt --quiet --potfile-disable 2>/dev/null | tail -1

echo "NTLM Performance:"
echo "aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0" > test_ntlm.txt
timeout 30s hashcat -m 1000 test_ntlm.txt rockyou.txt --quiet --potfile-disable 2>/dev/null | tail -1

# Cleanup
rm -f test_*.txt

echo "✅ Benchmark complete!"
EOF
chmod +x /workspace/hashmancer/benchmark.sh

echo "✅ High Performance setup complete!"
echo ""
echo "🔥 SYSTEM OPTIMIZED FOR MAXIMUM PERFORMANCE 🔥"
echo "================================================"
echo "📚 Wordlists: $(ls -1 /workspace/hashmancer/wordlists/*.txt | wc -l) comprehensive lists available"
echo "⚙️ Rules: $(ls -1 /workspace/hashmancer/rules/*.rule | wc -l) rule sets loaded"
echo "🎯 GPU: Maximum performance mode enabled"
echo "💻 CPU: Performance governor active"
echo "📊 Monitoring: ./advanced_monitor.sh"
echo "🏃 Benchmark: ./benchmark.sh"
echo ""
echo "💡 Ready for high-speed hash cracking!"