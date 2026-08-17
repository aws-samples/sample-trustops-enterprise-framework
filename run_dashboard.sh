#!/bin/bash
# Quick start script for the TrustOps Streamlit Dashboard (techtalk demo).
# Handles all the demo environment prep so the dashboard "just works":
#   - unsets the shadowing AWS_REGION (stack lives in us-east-1)
#   - loads .env so the CLI/clients see the real resource names
#   - activates the project venv
#
# Usage:  ./run_dashboard.sh
set -euo pipefail

# Always run from the repo root (the directory this script lives in).
cd "$(dirname "$0")"

echo "🎯 TrustOps Dashboard Launcher"
echo "=============================="
echo ""

# 1. Fix region shadowing — a shell-exported AWS_REGION would override the
#    region your stack was deployed to. Let .env be the single source of truth.
unset AWS_REGION AWS_DEFAULT_REGION

# 2. Load demo resource names into the environment BEFORE Python imports run,
#    otherwise the app falls back to default table/bucket names.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "✓ Loaded .env (region=${AWS_REGION:-unset})"
else
    echo "⚠️  No .env found at repo root — dashboard may not find stack resources."
fi

# 3. Activate the project virtualenv (python3.11 with all deps + botocore[crt]).
if [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "✓ Activated venv"
else
    echo "⚠️  No venv found — using system Python."
fi

# 4. Make sure Streamlit is available.
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed in this environment."
    echo "   Install with:  pip install streamlit plotly pandas"
    exit 1
fi

echo ""
echo "🚀 Launching TrustOps Dashboard..."
echo "   URL: http://localhost:8501   (Ctrl+C to stop)"
echo ""

# 5. Launch the real dashboard entry point.
exec streamlit run dashboard/app.py "$@"
