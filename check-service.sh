#!/bin/bash
# Diagnostic script to check AI service status

echo "=== AI Service Diagnostic ==="
echo ""

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "❌ PM2 is not installed"
    echo "   Install with: npm install -g pm2"
    exit 1
fi

echo "✓ PM2 is installed"
echo ""

# Check PM2 status
echo "=== PM2 Process Status ==="
pm2 status
echo ""

# Check if siha-ai is running
if pm2 list | grep -q "siha-ai.*online"; then
    echo "✓ siha-ai service is running"
else
    echo "❌ siha-ai service is NOT running"
    echo ""
    echo "To start the service:"
    echo "  cd /srv/siha/ai-service-python"
    echo "  pm2 start ecosystem.config.js"
    echo ""
fi

# Check if port 3001 is listening
echo "=== Port 3001 Status ==="
if sudo netstat -tlnp 2>/dev/null | grep -q ":3001" || sudo ss -tlnp 2>/dev/null | grep -q ":3001"; then
    echo "✓ Port 3001 is listening"
    sudo netstat -tlnp 2>/dev/null | grep ":3001" || sudo ss -tlnp 2>/dev/null | grep ":3001"
else
    echo "❌ Port 3001 is NOT listening"
    echo "   The service may not be running or not bound to port 3001"
fi
echo ""

# Test health endpoint
echo "=== Health Check ==="
if curl -s http://localhost:3001/health > /dev/null; then
    echo "✓ Health endpoint is responding"
    curl -s http://localhost:3001/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:3001/health
else
    echo "❌ Health endpoint is NOT responding"
    echo "   Service may not be running or not accessible"
fi
echo ""

# Check recent logs
echo "=== Recent Error Logs (last 10 lines) ==="
if [ -f ~/.pm2/logs/siha-ai-error.log ]; then
    tail -10 ~/.pm2/logs/siha-ai-error.log
else
    echo "No error log file found"
fi
echo ""

# Check memory
echo "=== System Memory ==="
free -h
echo ""

# Check swap
echo "=== Swap Status ==="
swapon --show || echo "No swap configured"
echo ""

# Check if service directory exists
echo "=== Service Directory ==="
if [ -d "/srv/siha/ai-service-python" ]; then
    echo "✓ Service directory exists: /srv/siha/ai-service-python"
    if [ -f "/srv/siha/ai-service-python/run.sh" ]; then
        echo "✓ run.sh exists"
    else
        echo "❌ run.sh not found"
    fi
    if [ -f "/srv/siha/ai-service-python/ecosystem.config.js" ]; then
        echo "✓ ecosystem.config.js exists"
    else
        echo "❌ ecosystem.config.js not found"
    fi
else
    echo "❌ Service directory not found: /srv/siha/ai-service-python"
    echo "   Update ecosystem.config.js with correct path"
fi
echo ""

echo "=== Quick Fix Commands ==="
echo ""
echo "1. Start the service:"
echo "   cd /srv/siha/ai-service-python"
echo "   pm2 start ecosystem.config.js"
echo ""
echo "2. Restart the service:"
echo "   pm2 restart siha-ai"
echo ""
echo "3. View logs:"
echo "   pm2 logs siha-ai"
echo ""
echo "4. Monitor in real-time:"
echo "   pm2 monit"
echo ""

