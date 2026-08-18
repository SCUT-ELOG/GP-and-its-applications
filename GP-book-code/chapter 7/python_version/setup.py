#!/usr/bin/env python
"""
安装依赖并运行快速测试
"""

import subprocess
import sys
import os

def install_dependencies():
    """安装依赖包"""
    print("="*70)
    print("Installing dependencies...")
    print("="*70)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies!")
        return False

def run_quick_test():
    """运行快速测试"""
    print("\n" + "="*70)
    print("Running quick test...")
    print("="*70)
    
    try:
        # 修改参数以加快测试
        print("\nNote: Using reduced parameters for quick testing")
        print("  - Population: 100 (default 1000)")
        print("  - Generations: 200 (default 2000)")
        print("\nRunning test on MONK-1 dataset...\n")
        
        subprocess.check_call([sys.executable, "test_simple.py"])
        print("\n✅ Test completed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Test failed!")
        return False

def main():
    print("="*70)
    print("GEP Classification - Python Version Setup")
    print("="*70)
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        sys.exit(1)
    
    print(f"\n✅ Python version: {sys.version.split()[0]}")
    
    # 询问是否安装依赖
    response = input("\nInstall dependencies? (y/n, default: y): ").strip().lower()
    
    if response != 'n':
        if not install_dependencies():
            sys.exit(1)
    
    # 询问是否运行测试
    response = input("\nRun quick test? (y/n, default: y): ").strip().lower()
    
    if response != 'n':
        if not run_quick_test():
            sys.exit(1)
    
    print("\n" + "="*70)
    print("Setup completed!")
    print("="*70)
    print("\nYou can now run:")
    print("  python main.py          - Interactive mode")
    print("  python test_simple.py   - Quick test")
    print("\nFor more information, see:")
    print("  README.md")
    print("  QUICK_START.md")
    print("  CONVERSION_SUMMARY.md")

if __name__ == "__main__":
    main()
