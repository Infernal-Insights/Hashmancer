#!/bin/bash
# Quick installer for Hashmancer CLI command

set -e

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a wrapper script for hashmancer command
WRAPPER_SCRIPT="#!/bin/bash
cd \"$PROJECT_DIR\"
export PYTHONPATH=\"$PROJECT_DIR:\$PYTHONPATH\"
python -m hashmancer.cli.main \"\$@\"
"

# Determine installation directory
if [ -w "/usr/local/bin" ]; then
    INSTALL_DIR="/usr/local/bin"
elif [ -w "$HOME/.local/bin" ]; then
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
else
    echo "Creating ~/.local/bin directory..."
    mkdir -p "$HOME/.local/bin"
    INSTALL_DIR="$HOME/.local/bin"
fi

HASHMANCER_BIN="$INSTALL_DIR/hashmancer"

echo "Installing hashmancer CLI to $HASHMANCER_BIN"

# Write the wrapper script
echo "$WRAPPER_SCRIPT" > "$HASHMANCER_BIN"
chmod +x "$HASHMANCER_BIN"

echo "✅ Hashmancer CLI installed successfully!"
echo "🔧 Installation location: $HASHMANCER_BIN"

# Check if the install directory is in PATH
if [[ ":$PATH:" == *":$INSTALL_DIR:"* ]]; then
    echo "✅ $INSTALL_DIR is in your PATH"
    echo "🚀 You can now run: hashmancer --help"
else
    echo "⚠️  $INSTALL_DIR is not in your PATH"
    echo "Add this to your ~/.bashrc or ~/.zshrc:"
    echo "export PATH=\"$INSTALL_DIR:\$PATH\""
    echo ""
    echo "Or run this command now:"
    echo "echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
fi

echo ""
echo "🧪 Testing installation..."
if command -v hashmancer >/dev/null 2>&1; then
    echo "✅ hashmancer command is available!"
    hashmancer --version
else
    echo "⚠️  hashmancer not found in PATH. Add $INSTALL_DIR to your PATH and try again."
fi

echo ""
echo "🎯 Quick test commands:"
echo "  hashmancer --help"
echo "  hashmancer sshkey --help"
echo "  hashmancer sshkey setup"