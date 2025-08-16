#!/bin/bash
# Hashmancer Vast.ai Demo Script
# Demonstrates automatic worker deployment

echo "🔓 Hashmancer Vast.ai Worker Auto-Setup Demo"
echo "============================================="

# Check if CLI is working
if ! ./hashmancer-cli --version > /dev/null 2>&1; then
    echo "❌ Hashmancer CLI not found or not working"
    exit 1
fi

echo "✅ Hashmancer CLI is ready"

# Demo 1: Show server commands
echo ""
echo "📋 Available Server Commands:"
echo "./hashmancer-cli server start          # Start Hashmancer server"
echo "./hashmancer-cli server status         # Check server status" 
echo "./hashmancer-cli server host-setup     # Host worker setup script"
echo "./hashmancer-cli server discover       # Find local workers"
echo "./hashmancer-cli server add-worker     # Deploy Vast.ai workers"
echo "./hashmancer-cli server jobs           # List jobs"
echo "./hashmancer-cli server delete-jobs    # Delete jobs in batch"

# Demo 2: Show worker commands  
echo ""
echo "👷 Available Worker Commands:"
echo "./hashmancer-cli worker start          # Start local worker"
echo "./hashmancer-cli worker status --all   # Check all workers"
echo "./hashmancer-cli worker logs <id>      # View worker logs"
echo "./hashmancer-cli worker skip <id>      # Skip current job"
echo "./hashmancer-cli worker stop <id>      # Stop worker"

# Demo 3: Show Vast.ai workflow
echo ""
echo "🚀 Vast.ai Automatic Deployment Workflow:"
echo ""
echo "1. Start your server:"
echo "   ./hashmancer-cli server start --host 0.0.0.0 --port 8080"
echo ""
echo "2. Host the setup script:"
echo "   ./hashmancer-cli server host-setup --port 8888 &"
echo ""
echo "3. Deploy workers automatically:"
echo "   export VAST_API_KEY=your_vast_api_key"
echo "   ./hashmancer-cli server add-worker 3080 --count 2 --max-price 0.75"
echo ""
echo "4. Monitor workers:"
echo "   ./hashmancer-cli worker status --all"
echo "   ./hashmancer-cli server discover"

# Demo 4: Show hashes.com integration
echo ""
echo "🌐 Hashes.com Integration:"
echo "./hashmancer-cli hashes pull --api-key YOUR_KEY --exclude-md5 --exclude-btc"
echo "./hashmancer-cli hashes watch --api-key YOUR_KEY --interval 300"
echo "./hashmancer-cli hashes stats --api-key YOUR_KEY"

# Demo 5: Show darkling usage
echo ""
echo "⚡ Direct Hash Cracking (Darkling):"
echo "./hashmancer-cli darkling crack hashes.txt wordlist.txt -m 1000 -a 0"
echo "./hashmancer-cli darkling mask hashes.txt '?u?l?l?l?l?l?d?d' -m 1000"
echo "./hashmancer-cli darkling benchmark"
echo "./hashmancer-cli darkling list-algorithms"

echo ""
echo "📖 Complete documentation available in VAST_AI_SETUP.md"
echo ""
echo "🎯 Example: Deploy 3 RTX 3080 workers under $0.60/hr:"
echo "   export VAST_API_KEY=your_key"
echo "   ./hashmancer-cli server host-setup --port 8888 &"
echo "   ./hashmancer-cli server add-worker 3080 --count 3 --max-price 0.60"
echo ""
echo "The workers will automatically:"
echo "  • Download and build the Docker image"
echo "  • Find your server IP address" 
echo "  • Register with your server"
echo "  • Start processing jobs"
echo ""
echo "✨ Completely hands-free setup! ✨"