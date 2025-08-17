#!/usr/bin/env python3
"""
Test script to verify Opus integration is ready
"""

import os
import sys
from pathlib import Path

def test_dependencies():
    """Test that all required dependencies are installed"""
    print("🔍 Testing Dependencies...")
    
    try:
        import anthropic
        print("  ✅ anthropic")
    except ImportError:
        print("  ❌ anthropic - run: pip install anthropic")
        return False
    
    try:
        import github
        print("  ✅ PyGithub")
    except ImportError:
        print("  ❌ PyGithub - run: pip install PyGithub")
        return False
    
    try:
        import matplotlib
        print("  ✅ matplotlib")
    except ImportError:
        print("  ❌ matplotlib - run: pip install matplotlib")
        return False
    
    try:
        import psutil
        print("  ✅ psutil")
    except ImportError:
        print("  ❌ psutil - run: pip install psutil")
        return False
    
    return True

def test_infrastructure():
    """Test that infrastructure files are in place"""
    print("\n🏗️ Testing Infrastructure...")
    
    required_files = [
        "scripts/opus-client.py",
        "scripts/opus-integration-starter.py", 
        "DEVELOPMENT_ISSUES_TRACKER.md",
        "OPUS_INTEGRATION_WORKFLOW.md",
        "EXECUTIVE_SUMMARY.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_security_fixes():
    """Test that security fixes were applied"""
    print("\n🔒 Testing Security Fixes...")
    
    # Test Docker security fix
    try:
        with open("Dockerfile.worker", "r") as f:
            content = f.read()
        if "USER hashmancer" in content:
            print("  ✅ Docker worker runs as non-root user")
        else:
            print("  ❌ Docker worker still runs as root")
            return False
    except FileNotFoundError:
        print("  ❌ Dockerfile.worker not found")
        return False
    
    # Test Redis upgrade
    try:
        with open("pyproject.toml", "r") as f:
            content = f.read()
        if "redis>=7.0.8" in content:
            print("  ✅ Redis dependency upgraded to secure version")
        else:
            print("  ❌ Redis dependency not upgraded")
            return False
    except FileNotFoundError:
        print("  ❌ pyproject.toml not found")
        return False
    
    # Test auth security markers
    try:
        with open("hashmancer/server/auth_middleware.py", "r") as f:
            content = f.read()
        if "TODO: Security Review Required" in content:
            print("  ✅ Security review markers added to auth middleware")
        else:
            print("  ❌ Security review markers not found")
            return False
    except FileNotFoundError:
        print("  ❌ auth_middleware.py not found")
        return False
    
    return True

def test_environment():
    """Test environment setup"""
    print("\n🌍 Testing Environment...")
    
    # Check for API key availability
    if os.getenv("ANTHROPIC_API_KEY"):
        print("  ✅ ANTHROPIC_API_KEY found in environment")
        api_key_ready = True
    else:
        print("  ⚠️ ANTHROPIC_API_KEY not set - add to environment or .env file")
        api_key_ready = False
    
    # Check for GitHub token
    if os.getenv("GITHUB_TOKEN"):
        print("  ✅ GITHUB_TOKEN found in environment")
    else:
        print("  ⚠️ GITHUB_TOKEN not set - needed for automated PRs")
    
    # Check .env.example exists
    if Path(".env.example").exists():
        print("  ✅ .env.example template available")
    else:
        print("  ❌ .env.example template missing")
    
    return api_key_ready

def test_opus_connection():
    """Test connection to Opus API"""
    print("\n🤖 Testing Opus Connection...")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ⚠️ Skipping - ANTHROPIC_API_KEY not set")
        return False
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        # Simple test message
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Use Haiku for testing
            max_tokens=100,
            messages=[{
                "role": "user", 
                "content": "Reply with exactly: 'Opus integration test successful'"
            }]
        )
        
        if "test successful" in response.content[0].text.lower():
            print("  ✅ Claude API connection successful")
            return True
        else:
            print("  ❌ Unexpected response from Claude API")
            return False
            
    except Exception as e:
        print(f"  ❌ Claude API connection failed: {e}")
        return False

def show_next_steps():
    """Show next steps for user"""
    print("\n🎯 NEXT STEPS:")
    print("=" * 50)
    
    print("\n1. 🔑 SET UP API KEYS:")
    print("   - Get your Anthropic API key from: https://console.anthropic.com/")
    print("   - Add to environment: export ANTHROPIC_API_KEY=your_key_here")
    print("   - Or create .env file from .env.example template")
    
    print("\n2. 🧪 RUN FIRST ANALYSIS:")
    print("   - Basic test: python3 scripts/opus-client.py")
    print("   - Security analysis: python3 scripts/opus-client.py --analysis-type security")
    
    print("\n3. 🚀 DEPLOY AUTOMATION:")
    print("   - Set up GitHub Actions with your API keys in secrets")
    print("   - Enable automated daily analysis")
    print("   - Review generated PRs and merge approved fixes")
    
    print("\n4. 📚 READ DOCUMENTATION:")
    print("   - EXECUTIVE_SUMMARY.md - Strategic overview")
    print("   - OPUS_INTEGRATION_WORKFLOW.md - Complete automation guide")
    print("   - DEVELOPMENT_ISSUES_TRACKER.md - All identified issues")

def main():
    """Main test function"""
    print("🧪 OPUS INTEGRATION READINESS TEST")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    # Run all tests
    if test_dependencies():
        tests_passed += 1
    
    if test_infrastructure():
        tests_passed += 1
    
    if test_security_fixes():
        tests_passed += 1
    
    if test_environment():
        tests_passed += 1
    
    if test_opus_connection():
        tests_passed += 1
    
    # Show results
    print(f"\n📊 TEST RESULTS: {tests_passed}/{total_tests} PASSED")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - OPUS INTEGRATION READY!")
        print("\n✅ You can now run automated analysis and fixes!")
    elif tests_passed >= 3:
        print("⚠️ MOSTLY READY - Fix remaining issues for full automation")
    else:
        print("❌ SETUP INCOMPLETE - Follow next steps below")
    
    show_next_steps()
    
    return tests_passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)