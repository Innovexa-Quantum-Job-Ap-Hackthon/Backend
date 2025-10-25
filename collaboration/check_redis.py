#!/usr/bin/env python3
"""
Redis Setup and Health Check Script
This script checks if Redis is installed and running, and provides setup instructions if needed.
"""

import subprocess
import sys
import platform

def check_redis_installation():
    """Check if Redis is installed"""
    try:
        result = subprocess.run(['redis-cli', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Redis CLI is installed")
            return True
        else:
            print("❌ Redis CLI not found")
            return False
    except FileNotFoundError:
        print("❌ Redis CLI not found")
        return False

def check_redis_server():
    """Check if Redis server is running"""
    try:
        result = subprocess.run(['redis-cli', 'ping'], capture_output=True, text=True)
        if result.returncode == 0 and 'PONG' in result.stdout:
            print("✅ Redis server is running")
            return True
        else:
            print("❌ Redis server is not running")
            return False
    except FileNotFoundError:
        print("❌ Redis CLI not available")
        return False

def install_redis_windows():
    """Provide Redis installation instructions for Windows"""
    print("\n🔧 Redis Installation Instructions for Windows:")
    print("1. Download Redis for Windows from: https://redis.io/download")
    print("2. Or use Chocolatey: choco install redis-64")
    print("3. Or use Winget: winget install Redis.Redis")
    print("4. Start Redis server: redis-server")
    print("5. Test connection: redis-cli ping")

def install_redis_linux():
    """Provide Redis installation instructions for Linux"""
    print("\n🔧 Redis Installation Instructions for Linux:")
    print("Ubuntu/Debian:")
    print("  sudo apt update")
    print("  sudo apt install redis-server")
    print("  sudo systemctl start redis-server")
    print("  sudo systemctl enable redis-server")
    print("")
    print("CentOS/RHEL:")
    print("  sudo yum install redis")
    print("  sudo systemctl start redis")
    print("  sudo systemctl enable redis")

def install_redis_macos():
    """Provide Redis installation instructions for macOS"""
    print("\n🔧 Redis Installation Instructions for macOS:")
    print("Using Homebrew:")
    print("  brew install redis")
    print("  brew services start redis")
    print("")
    print("Or using MacPorts:")
    print("  sudo port install redis")
    print("  sudo port load redis")

def start_redis_server():
    """Try to start Redis server"""
    try:
        print("\n🚀 Attempting to start Redis server...")
        if platform.system() == "Windows":
            subprocess.run(['redis-server'], check=True)
        else:
            subprocess.run(['sudo', 'systemctl', 'start', 'redis-server'], check=True)
        print("✅ Redis server started successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to start Redis server automatically")
        return False
    except FileNotFoundError:
        print("❌ Redis server executable not found")
        return False

def main():
    print("🔍 Checking Redis Installation and Status...\n")

    os_name = platform.system()
    print(f"Operating System: {os_name}")

    # Check if Redis CLI is installed
    redis_installed = check_redis_installation()

    if not redis_installed:
        print(f"\n❌ Redis is not installed on your {os_name} system.")
        if os_name == "Windows":
            install_redis_windows()
        elif os_name == "Linux":
            install_redis_linux()
        elif os_name == "Darwin":  # macOS
            install_redis_macos()
        else:
            print(f"❓ Please check Redis installation instructions for {os_name}")
        return

    # Check if Redis server is running
    redis_running = check_redis_server()

    if not redis_running:
        print("\n❌ Redis server is not running.")
        print("Attempting to start Redis server...")

        if not start_redis_server():
            print("\n🔧 Manual Redis Server Start Instructions:")
            if os_name == "Windows":
                print("  redis-server")
            else:
                print("  sudo systemctl start redis-server")
                print("  # Or: sudo service redis-server start")
        return

    print("\n🎉 Redis is properly installed and running!")
    print("✅ Your Quantum Dashboard Redis caching is ready to use.")

if __name__ == "__main__":
    main()
