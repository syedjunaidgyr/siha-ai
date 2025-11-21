# t2.micro (Free Tier) EC2 Setup Guide

This guide is specifically optimized for **t2.micro instances** with only **1GB RAM**. This is a very constrained environment that requires aggressive memory management.

## ⚠️ Important Limitations

**t2.micro instances have:**
- 1 vCPU
- 1GB RAM (very limited!)
- Burstable CPU performance
- Free tier eligible (750 hours/month)

**This setup will work but with limitations:**
- Slower processing times
- More frequent service restarts
- May struggle with very large video payloads
- Consider upgrading to t3.micro (2GB RAM) or t3.small (2GB RAM) for better performance

## Step 1: Create Swap Space (CRITICAL for 1GB RAM)

Swap space is **essential** for t2.micro. Without it, the service will crash frequently.

```bash
# Check current memory and swap
free -h

# Create 2GB swap file (recommended for t2.micro)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify swap is active
swapon --show
free -h

# Make swap permanent (survives reboots)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Optimize swap usage (prefer RAM, use swap only when needed)
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**Expected output after setup:**
```
              total        used        free      shared  buff/cache   available
Mem:          985Mi       200Mi       150Mi        10Mi       634Mi       700Mi
Swap:         2.0Gi         0Mi       2.0Gi
```

## Step 2: Optimize System for Low Memory

```bash
# Reduce system memory usage
# Disable unnecessary services (adjust based on your needs)
sudo systemctl disable --now postfix 2>/dev/null || true  # Mail service
sudo systemctl disable --now sendmail 2>/dev/null || true

# Limit log file sizes to prevent disk/memory issues
sudo journalctl --vacuum-size=100M

# Set up log rotation for PM2
sudo nano /etc/logrotate.d/pm2
```

Add this content to `/etc/logrotate.d/pm2`:
```
/home/ec2-user/.pm2/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 ec2-user ec2-user
}
```

## Step 3: Increase System Limits

```bash
# Edit limits configuration
sudo nano /etc/security/limits.conf

# Add these lines at the end:
* soft nofile 4096
* hard nofile 8192
* soft nproc 2048
* hard nproc 4096

# For systemd (PM2)
sudo nano /etc/systemd/system.conf

# Uncomment and set:
DefaultLimitNOFILE=4096
DefaultLimitNPROC=2048

# Reload systemd
sudo systemctl daemon-reload

# Log out and back in for limits to take effect
```

## Step 4: Install and Configure Service

```bash
# Navigate to service directory
cd /srv/siha/ai-service-python

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (this may take a while on t2.micro)
pip install --upgrade pip
pip install -r requirements.txt

# Make run script executable
chmod +x run.sh
```

## Step 5: Configure PM2

The configuration is already optimized for t2.micro in `ecosystem.config.js`:
- `max_memory_restart: '400M'` - Restarts before using too much RAM
- `--max-requests 5` - Very frequent worker restarts
- Reduced worker connections

```bash
# Start the service
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Setup auto-start on reboot
pm2 startup
# Follow the instructions shown
```

## Step 6: Monitor Memory Usage

```bash
# Real-time monitoring
pm2 monit

# Check system memory
watch -n 1 free -h

# Check process memory
ps aux --sort=-%mem | head -10

# Check swap usage
swapon --show
```

## Step 7: Security Group Configuration

```bash
# Ensure port 3001 is only accessible from your backend
# Via AWS Console:
# EC2 → Security Groups → Your SG → Inbound Rules
# Add: Custom TCP, Port 3001, Source: Backend Security Group
```

## Performance Expectations

**With t2.micro (1GB RAM):**
- Processing time: 5-15 seconds per video analysis (slower than larger instances)
- Concurrent requests: 1-2 at a time (limited by memory)
- Service restarts: Every 5 requests (to prevent memory leaks)
- May experience occasional slowdowns during CPU burst credits exhaustion

## Troubleshooting

### Service Keeps Crashing

```bash
# Check if swap is active
free -h
swapon --show

# If no swap, create it (see Step 1)

# Check OOM kills
dmesg | grep -i "out of memory"
journalctl -k | grep -i oom

# Check PM2 logs
pm2 logs siha-ai --err --lines 100
```

### High Memory Usage

```bash
# Check what's using memory
ps aux --sort=-%mem | head -20

# Restart service to free memory
pm2 restart siha-ai

# If still high, reduce max_memory_restart to 300M in ecosystem.config.js
```

### Slow Performance

```bash
# Check CPU burst credits (t2 instances have burstable performance)
# Install CloudWatch agent or check via AWS Console
# If credits exhausted, instance will throttle

# Check system load
uptime
top

# Consider:
# 1. Reducing frame count in requests (backend should limit to 8-10 frames)
# 2. Upgrading to t3.micro (2GB RAM) or t3.small
```

### Connection Refused (ECONNREFUSED 127.0.0.1:3001)

This error means the AI service is not running. Follow these steps:

```bash
# 1. Check if service is running
pm2 status

# 2. If not running, start it
cd /srv/siha/ai-service-python
pm2 start ecosystem.config.js

# 3. Verify it's listening on port 3001
sudo netstat -tlnp | grep 3001
# or
sudo ss -tlnp | grep 3001

# 4. Test health endpoint
curl http://localhost:3001/health

# 5. Check logs for errors
pm2 logs siha-ai --err --lines 50

# 6. If service won't start, check:
# - Virtual environment is activated in run.sh
# - Dependencies are installed: pip install -r requirements.txt
# - Python version: python3 --version (should be 3.11+)
# - Disk space: df -h
# - Memory: free -h (ensure swap is configured)
```

**Quick diagnostic script:**
```bash
cd /srv/siha/ai-service-python
chmod +x check-service.sh
./check-service.sh
```

## Optimization Tips

1. **Reduce Frame Count**: Process fewer frames per request (backend should limit to 8-10 frames max)
2. **Enable Compression**: Backend should compress frames before sending (already implemented)
3. **Use S3**: For large payloads, use S3 upload URLs (already implemented)
4. **Monitor Closely**: Use `pm2 monit` regularly to catch memory issues early
5. **Upgrade When Possible**: t3.micro (2GB) or t3.small (2GB) provide much better performance

## Cost Considerations

**Free Tier:**
- t2.micro: 750 hours/month free
- S3: 5GB storage, 20,000 GET requests, 2,000 PUT requests free
- Data transfer: 15GB out free

**If you exceed free tier:**
- t2.micro: ~$8.50/month
- t3.micro: ~$8.50/month (2GB RAM - much better!)
- t3.small: ~$17/month (2GB RAM, better CPU)

**Recommendation**: If you're paying, upgrade to t3.micro for the same price but 2x RAM.

## Quick Commands Reference

```bash
# Service management
pm2 status
pm2 restart siha-ai
pm2 logs siha-ai
pm2 monit

# Memory monitoring
free -h
swapon --show
ps aux --sort=-%mem | head -10

# Health check
curl http://localhost:3001/health

# Check swap
free -h
cat /proc/swaps

# Check OOM kills
dmesg | grep -i oom
journalctl -k | grep -i oom
```

## When to Upgrade

Consider upgrading from t2.micro if you experience:
- Frequent OOM kills even with swap
- Processing times > 20 seconds consistently
- Need to handle > 2 concurrent requests
- Service restarts too frequently (> every 5 requests)

**Recommended upgrade path:**
1. t2.micro (1GB) → t3.micro (2GB) - Same price, 2x RAM
2. t3.micro (2GB) → t3.small (2GB) - Better CPU, more burst credits
3. t3.small → t3.medium (4GB) - For production workloads

