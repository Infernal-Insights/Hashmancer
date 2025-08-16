#!/bin/bash
# Test DigitalOcean Setup Readiness

echo "🌊 DigitalOcean Hashmancer Setup Test"
echo "====================================="

# Check if setup script exists
if [ -f "scripts/setup-digitalocean-server.sh" ]; then
    echo "✅ Setup script ready"
    echo "   Location: scripts/setup-digitalocean-server.sh"
else
    echo "❌ Setup script missing"
fi

# Check if management script exists
if [ -f "scripts/manage-server.sh" ]; then
    echo "✅ Management script ready"
    echo "   Location: scripts/manage-server.sh"
else
    echo "❌ Management script missing"
fi

# Show what the scripts will do
echo ""
echo "📋 Setup Process Overview:"
echo "1. ✅ Update Ubuntu system packages"
echo "2. ✅ Configure UFW firewall (ports 22, 80, 443, 8080)"
echo "3. ✅ Install Python 3, Redis, Nginx"
echo "4. ✅ Create 'hashmancer' user and directories"
echo "5. ✅ Install Docker (for testing)"
echo "6. ✅ Configure Redis for Hashmancer"
echo "7. ✅ Set up Python virtual environment"
echo "8. ✅ Install Hashmancer server code"
echo "9. ✅ Create systemd service"
echo "10. ✅ Configure Nginx reverse proxy"
echo "11. ✅ Start all services"
echo "12. ✅ Test server accessibility"

echo ""
echo "🎯 After Setup, Your Server Will:"
echo "• 🌐 Be accessible at http://YOUR_DROPLET_IP:8080"
echo "• 🏥 Provide health checks at /health endpoint"
echo "• 👷 Accept worker registrations at /worker/register"
echo "• 📊 Show connected workers at /workers"
echo "• 🔒 Be protected by UFW firewall"
echo "• 🔄 Auto-restart if it crashes"

echo ""
echo "📝 To Deploy on DigitalOcean:"
echo "1. Create Ubuntu 22.04 droplet (2GB RAM minimum)"
echo "2. SSH to your droplet: ssh root@YOUR_DROPLET_IP"
echo "3. Upload setup script: scp scripts/setup-digitalocean-server.sh root@YOUR_DROPLET_IP:/tmp/"
echo "4. Run setup: cd /tmp && chmod +x setup-digitalocean-server.sh && ./setup-digitalocean-server.sh"
echo "5. Test: curl http://YOUR_DROPLET_IP:8080/health"

echo ""
echo "🎮 For Vast.ai Workers:"
echo "• Set environment variable: HASHMANCER_SERVER_IP=YOUR_DROPLET_IP"
echo "• Workers will automatically connect to port 8080"
echo "• Use Docker image: yourusername/hashmancer-worker:latest"

echo ""
echo "⚙️  Management Commands (after setup):"
echo "• sudo /opt/hashmancer/scripts/manage-server.sh status"
echo "• sudo /opt/hashmancer/scripts/manage-server.sh restart"
echo "• sudo /opt/hashmancer/scripts/manage-server.sh logs"
echo "• sudo /opt/hashmancer/scripts/manage-server.sh workers"
echo "• sudo /opt/hashmancer/scripts/manage-server.sh monitor"

echo ""
echo "✨ Ready to deploy your DigitalOcean Hashmancer server!"