#!/usr/bin/env python3
"""
Quick launcher for TrustOps Streamlit Dashboard

Usage:
    python run_dashboard.py
"""

import subprocess  # nosec B404 - fixed, hardcoded argv only; no shell, no user input
import sys
from pathlib import Path


def check_streamlit_installed():
    """Check if streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def install_dependencies():
    """Install required dashboard dependencies."""
    print("📦 Installing dashboard dependencies...")
    subprocess.check_call([  # nosec B603 - fixed argv, no shell
        sys.executable, "-m", "pip", "install",
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
        "pandas>=2.1.0"
    ])
    print("✓ Dependencies installed\n")


def check_demo_results():
    """Check if demo results exist."""
    results_dir = Path("demo/results")
    if not results_dir.exists() or not any(results_dir.iterdir()):
        print("⚠️  No demo results found in demo/results/\n")
        print("Run the demo first to generate results:")
        print("  python demo/run_demo.py\n")
        response = input("Continue anyway? (y/n): ")
        if not response.lower().startswith('y'):
            sys.exit(0)


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("🚀 Launching TrustOps Dashboard...\n")
    print("Dashboard will open at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard\n")
    
    try:
        subprocess.run([  # nosec B603 - fixed argv, no shell
            sys.executable, "-m", "streamlit", "run", "dashboard/app.py"
        ], check=False)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped")


def main():
    """Main launcher function."""
    print("🎯 TrustOps Dashboard Launcher")
    print("=" * 50)
    print()
    
    # Check and install dependencies
    if not check_streamlit_installed():
        print("❌ Streamlit is not installed.\n")
        response = input("Install required dependencies? (y/n): ")
        if response.lower().startswith('y'):
            install_dependencies()
        else:
            print("Cannot launch dashboard without dependencies.")
            sys.exit(1)
    
    # Check for demo results
    check_demo_results()
    
    # Launch dashboard
    launch_dashboard()


if __name__ == '__main__':
    main()
