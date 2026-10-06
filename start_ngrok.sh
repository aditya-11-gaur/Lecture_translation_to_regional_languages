#!/bin/bash

echo "Starting NPTEL Pipeline Ngrok setup..."

# 1. Install ngrok if missing
if ! command -v ngrok &> /dev/null; then
    echo "Downloading and installing ngrok..."
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
    sudo apt update
    sudo apt install ngrok
else
    echo "✅ ngrok is already installed."
fi

# 2. Authenticate ngrok
# Load .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$NGROK_AUTHTOKEN" ]; then
    echo ""
    echo "⚠️  Ngrok Authtoken not found in .env"
    read -p "🔑 Please paste your Authtoken (from dashboard.ngrok.com): " INPUT_TOKEN
    echo "NGROK_AUTHTOKEN=\"$INPUT_TOKEN\"" >> .env
    NGROK_AUTHTOKEN=$INPUT_TOKEN
fi
ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null
echo "✅ ngrok authenticated."

# 3. Ensure Streamlit is running
if ! pgrep -f "streamlit run app.py" > /dev/null; then
    echo "⚠️  Streamlit is not running! Starting it in the background..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    nohup streamlit run app.py --server.port 8501 > streamlit.log 2>&1 &
    sleep 3
    echo "✅ Streamlit started."
else
    echo "✅ Streamlit is already running."
fi

# 4. Start the Tunnel
echo ""
echo "======================================================="
echo "🌐 Starting Ngrok Public Tunnel to localhost:8501"
echo "======================================================="
echo ""
ngrok http --domain=benito-nonimperative-improvingly.ngrok-free.dev 8501

