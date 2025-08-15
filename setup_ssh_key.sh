#!/bin/bash

echo "🔑 Hashmancer SSH Key Setup for Vast.ai"
echo "======================================"

# Check if SSH key already exists
if [ -f ~/.ssh/id_rsa ]; then
    echo "✅ SSH key already exists at ~/.ssh/id_rsa"
    echo "📋 Your public key:"
    cat ~/.ssh/id_rsa.pub
    echo ""
    echo "📝 Next steps:"
    echo "1. Copy the public key above"
    echo "2. Go to https://cloud.vast.ai/account/"
    echo "3. Paste it in the 'SSH Key' section"
    echo "4. Save the changes"
else
    echo "🔧 Generating new SSH key for Hashmancer..."
    ssh-keygen -t rsa -b 4096 -C "hashmancer@vast.ai" -f ~/.ssh/id_rsa -N ""
    
    echo "✅ SSH key generated successfully!"
    echo ""
    echo "📋 Your public key (copy this):"
    echo "================================"
    cat ~/.ssh/id_rsa.pub
    echo "================================"
    echo ""
    echo "📝 Setup instructions:"
    echo "1. Copy the public key above (the entire line starting with 'ssh-rsa')"
    echo "2. Go to https://cloud.vast.ai/account/"
    echo "3. Find the 'SSH Key' section"
    echo "4. Paste your public key there"
    echo "5. Click 'Save' or 'Update'"
fi

echo ""
echo "🧪 Testing SSH key setup..."

# Test if we can connect to vast.ai (this will fail until they add the key)
echo "💡 Once you've added the key to vast.ai, you can test connections with:"
echo "   ssh-keyscan vast.ai >> ~/.ssh/known_hosts"

# Check if vast.ai key is in known_hosts
if ! grep -q "vast.ai" ~/.ssh/known_hosts 2>/dev/null; then
    echo "🔧 Adding vast.ai to known hosts..."
    ssh-keyscan console.vast.ai >> ~/.ssh/known_hosts 2>/dev/null || true
fi

echo ""
echo "⚡ Advanced: Automatic Setup (requires vast-python)"
echo "If you want automatic setup, run:"
echo "   pip install vast-python"
echo "   vastai set ssh-key ~/.ssh/id_rsa.pub"

echo ""
echo "✅ SSH key setup complete!"
echo "🚀 After adding the key to vast.ai, you're ready to launch workers!"