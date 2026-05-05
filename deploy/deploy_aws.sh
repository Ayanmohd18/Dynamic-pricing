#!/bin/bash
# Olist Dynamic Pricing Engine - AWS Deployment Script

set -e

# --- 1. Environment Check ---
if [[ -z "$AWS_REGION" || -z "$S3_BUCKET" || -z "$EC2_INSTANCE_ID" || -z "$EC2_KEY_PATH" ]]; then
    echo "❌ Error: Missing required environment variables (AWS_REGION, S3_BUCKET, EC2_INSTANCE_ID, EC2_KEY_PATH)"
    exit 1
fi

EC2_IP=$(aws ec2 describe-instances --instance-ids $EC2_INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
SSH_CMD="ssh -i $EC2_KEY_PATH -o StrictHostKeyChecking=no ubuntu@$EC2_IP"

echo "🚀 Starting Deployment to AWS Region: $AWS_REGION ($EC2_IP)"

# --- 2. Sync Artifacts ---
echo "📦 Syncing model artifacts to S3..."
aws s3 sync models/ s3://$S3_BUCKET/models/ --exclude "*" --include "*.pkl" --include "*.json"

# --- 3. Remote Setup & Code Update ---
echo "📡 Updating code and dependencies on EC2..."
$SSH_CMD << EOF
    # Install dependencies if missing
    sudo apt-get update && sudo apt-get install -y python3.11 python3-pip nginx supervisor git redis-server
    
    # Clone or Pull repo
    if [ ! -d "DP" ]; then
        git clone https://github.com/user/DP.git
    fi
    cd DP
    git pull origin main
    
    # Python environment setup
    pip3 install -r configs/requirements.txt
    
    # Nginx Configuration (FastAPI Reverse Proxy)
    sudo bash -c 'cat > /etc/nginx/sites-available/pricing_engine <<NGINX
server {
    listen 80;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
NGINX'
    sudo ln -sf /etc/nginx/sites-available/pricing_engine /etc/nginx/sites-enabled/
    sudo systemctl restart nginx

    # Supervisor Configuration
    sudo bash -c 'cat > /etc/supervisor/conf.d/pricing.conf <<SUP
[program:fastapi]
command=python3.11 src/api/main.py
directory=/home/ubuntu/DP
autostart=true
autorestart=true
stderr_logfile=/var/log/fastapi.err.log
stdout_logfile=/var/log/fastapi.out.log

[program:streamlit]
command=streamlit run dashboard/streamlit_app.py --server.port 8501
directory=/home/ubuntu/DP
autostart=true
autorestart=true
SUP'
    sudo supervisorctl reread
    sudo supervisorctl update
    sudo supervisorctl restart all
EOF

# --- 4. Health Check & Rollback ---
echo "⏳ Waiting for services to stabilize (30s)..."
sleep 30

echo "🔍 Running Health Checks..."
if python3 deploy/health_check.py --url "http://$EC2_IP"; then
    echo "✅ DEPLOYMENT SUCCESSFUL"
else
    echo "🚨 HEALTH CHECK FAILED - ROLLING BACK..."
    $SSH_CMD "cd DP && git checkout HEAD@{1} && sudo supervisorctl restart all"
    echo "↩️ Rollback complete."
    exit 1
fi
