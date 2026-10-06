#!/bin/bash

echo "Starting NPTEL Pipeline deployment setup..."

# 1. Ensure Cloudflare Tunnel is installed
if ! command -v cloudflared &> /dev/null; then
    echo "Downloading and installing cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared-linux-amd64.deb
    rm cloudflared-linux-amd64.deb
else
    echo "cloudflared is already installed."
fi

# 2. Check if Streamlit is running
if ! pgrep -f "streamlit run app.py" > /dev/null; then
    echo "⚠️  Streamlit is not running! Starting it in the background..."
    # Source venv if it exists
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    # Set default password if not in .env
    if ! grep -q "APP_PASSWORD" .env 2>/dev/null; then
        echo "APP_PASSWORD=btpdemo" >> .env
        echo "Set default password 'btpdemo' in .env"
    fi
    nohup streamlit run app.py --server.port 8501 > streamlit.log 2>&1 &
    sleep 3
    echo "Streamlit started."
fi

# 3. Start Cloudflare Tunnel
echo ""
echo "======================================================="
echo "🌐 Starting Public Tunnel to localhost:8501"
echo "======================================================="
echo ""
cloudflared tunnel --url http://localhost:8501

