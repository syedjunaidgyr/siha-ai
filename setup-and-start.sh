#!/bin/bash
# Complete setup and start script for AI Service on EC2
# Run this after SSH'ing into your EC2 instance

set -e

echo "=========================================="
echo "AI Service Setup and Start Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as ec2-user
if [ "$USER" != "ec2-user" ]; then
    echo -e "${YELLOW}Warning: Not running as ec2-user. Some commands may need sudo.${NC}"
fi

# Step 1: Navigate to service directory
echo -e "${GREEN}[1/8]${NC} Checking service directory..."
SERVICE_DIR="/srv/siha/ai-service-python"

if [ ! -d "$SERVICE_DIR" ]; then
    echo -e "${RED}Error: Service directory not found at $SERVICE_DIR${NC}"
    echo "Please update SERVICE_DIR in this script or create the directory."
    exit 1
fi

cd "$SERVICE_DIR"
echo "✓ Service directory: $SERVICE_DIR"
echo ""

# Step 2: Check/create swap space (critical for t2.micro)
echo -e "${GREEN}[2/8]${NC} Checking swap space..."
if swapon --show | grep -q "/swapfile"; then
    echo "✓ Swap space already configured"
    swapon --show
else
    echo -e "${YELLOW}Swap space not found. Creating 2GB swap file...${NC}"
    echo "This is critical for t2.micro (1GB RAM) instances."
    
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    
    # Make it permanent
    if ! grep -q "/swapfile" /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    
    # Optimize swap usage
    if ! grep -q "vm.swappiness=10" /etc/sysctl.conf; then
        echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
        echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
        sudo sysctl -p
    fi
    
    echo "✓ Swap space created and configured"
    swapon --show
fi
echo ""

# Step 3: Check Python version
echo -e "${GREEN}[3/8]${NC} Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo -e "${RED}Error: Python3 not found${NC}"
    echo "Installing Python 3.11..."
    sudo amazon-linux-extras install python3.11 -y || sudo yum install python3.11 -y
fi
echo ""

# Step 4: Setup virtual environment
echo -e "${GREEN}[4/8]${NC} Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Step 5: Install/upgrade dependencies
echo -e "${GREEN}[5/8]${NC} Installing dependencies..."
pip install --upgrade pip --quiet
echo "Installing requirements (this may take a few minutes)..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 6: Verify gunicorn
echo -e "${GREEN}[6/8]${NC} Checking gunicorn..."
if ! command -v gunicorn &> /dev/null; then
    echo "Installing gunicorn..."
    pip install gunicorn
fi
echo "✓ Gunicorn available: $(gunicorn --version)"
echo ""

# Step 7: Make scripts executable
echo -e "${GREEN}[7/8]${NC} Making scripts executable..."
chmod +x run.sh
if [ -f "check-service.sh" ]; then
    chmod +x check-service.sh
fi
echo "✓ Scripts are executable"
echo ""

# Step 8: Start/restart service with PM2
echo -e "${GREEN}[8/8]${NC} Starting service with PM2..."

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo -e "${RED}Error: PM2 not found${NC}"
    echo "Installing PM2..."
    npm install -g pm2
fi

# Stop existing service if running
pm2 stop siha-ai 2>/dev/null || true
pm2 delete siha-ai 2>/dev/null || true

# Start the service
echo "Starting AI service..."
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

echo ""
echo "=========================================="
echo "Service Status"
echo "=========================================="
pm2 status
echo ""

# Wait a moment for service to start
sleep 3

# Test health endpoint
echo "=========================================="
echo "Health Check"
echo "=========================================="
if curl -s http://localhost:3001/health > /dev/null; then
    echo -e "${GREEN}✓ Service is responding!${NC}"
    echo ""
    echo "Health check response:"
    curl -s http://localhost:3001/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:3001/health
else
    echo -e "${RED}✗ Service is not responding${NC}"
    echo ""
    echo "Checking logs..."
    pm2 logs siha-ai --lines 20 --nostream
fi
echo ""

# Check if port is listening
echo "=========================================="
echo "Port Status"
echo "=========================================="
if sudo netstat -tlnp 2>/dev/null | grep -q ":3001" || sudo ss -tlnp 2>/dev/null | grep -q ":3001"; then
    echo -e "${GREEN}✓ Port 3001 is listening${NC}"
    sudo netstat -tlnp 2>/dev/null | grep ":3001" || sudo ss -tlnp 2>/dev/null | grep ":3001"
else
    echo -e "${RED}✗ Port 3001 is not listening${NC}"
fi
echo ""

# Display useful commands
echo "=========================================="
echo "Useful Commands"
echo "=========================================="
echo "View logs:        pm2 logs siha-ai"
echo "Restart service:  pm2 restart siha-ai"
echo "Stop service:     pm2 stop siha-ai"
echo "Monitor:          pm2 monit"
echo "Status:           pm2 status"
echo ""

# Setup auto-start on reboot (optional)
echo "=========================================="
echo "Auto-Start on Reboot"
echo "=========================================="
echo "To enable auto-start on reboot, run:"
echo "  pm2 startup"
echo "  (Then follow the instructions shown)"
echo ""

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Your AI service should now be running at:"
echo "  - Internal: http://localhost:3001"
echo "  - External: http://13.203.161.24:3001"
echo ""
echo "Backend should connect to: http://localhost:3001/api"

