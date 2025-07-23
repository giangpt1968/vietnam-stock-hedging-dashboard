#!/usr/bin/env python3
"""
Vietnam Stock Hedging Dashboard Launcher
=======================================

Simple launcher for the web dashboard with dependency management.
"""

import sys
import subprocess
import importlib

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("Please install manually: pip install streamlit plotly pandas numpy")
            return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        print("🚀 Launching Vietnam Stock Hedging Dashboard...")
        print("🌐 Dashboard will open in your browser")
        print("📊 Use Ctrl+C to stop the server")
        
        # Launch streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'vietnam_hedge_dashboard.py',
            '--server.port', '8501',
            '--server.headless', 'false'
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {str(e)}")
        print("💡 Try running manually: streamlit run vietnam_hedge_dashboard.py")

def main():
    """Main launcher function"""
    print("🇻🇳 Vietnam Stock Hedging Dashboard")
    print("=" * 50)
    
    # Check dependencies
    if not check_and_install_dependencies():
        return
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main() 