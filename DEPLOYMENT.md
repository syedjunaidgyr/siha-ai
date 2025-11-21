# AI Service Deployment Guide

> **For AWS EC2 deployment:**
> - **t2.micro (Free Tier, 1GB RAM)**: See [T2_MICRO_SETUP.md](./T2_MICRO_SETUP.md) for optimized configuration
> - **Other EC2 instances**: See [EC2_DEPLOYMENT.md](./EC2_DEPLOYMENT.md) for EC2-specific configurations

## Quick Start / Restart

### First Time Setup:
```bash
# SSH to your EC2 instance
ssh -i your-key.pem ec2-user@your-ec2-ip

# Navigate to service directory
cd /srv/siha/ai-service-python

# Make scripts executable
chmod +x run.sh check-service.sh

# Start the service
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Setup auto-start on reboot
pm2 startup
# Follow the instructions shown

# Verify it's running
pm2 status
curl http://localhost:3001/health
```

### Quick Restart (After Code Changes):
```bash
# SSH to server
cd /srv/siha/ai-service-python

# Restart with PM2
pm2 restart siha-ai

# Or if using ecosystem config
pm2 restart ecosystem.config.js

# Check status
pm2 status
pm2 logs siha-ai --lines 50
```

### If Service Won't Start (Connection Refused Error):
```bash
# Run diagnostic script
cd /srv/siha/ai-service-python
chmod +x check-service.sh
./check-service.sh

# Or manually check:
pm2 status
pm2 logs siha-ai --err
curl http://localhost:3001/health
```

## Using PM2 Ecosystem Config

1. **Update the path in `ecosystem.config.js`** to match your server directory:
   ```javascript
   cwd: '/srv/siha/ai-service-python', // Update this!
   ```

2. **Start the service:**
   ```bash
   pm2 start ecosystem.config.js
   ```

3. **Save PM2 configuration:**
   ```bash
   pm2 save
   pm2 startup  # Follow instructions to enable auto-start on reboot
   ```

## Manual Start (Alternative)

If not using PM2, you can start directly:

```bash
cd /srv/siha/ai-service-python
chmod +x run.sh
./run.sh
```

## Memory Management

The service is configured with aggressive memory management to prevent OOM kills. **Settings vary by instance size:**

### For t2.micro (1GB RAM) - Current Default:
- **1 worker process** (reduces memory usage)
- **Auto-restart after 5 requests** (very frequent to prevent memory leaks)
- **400MB memory limit** (PM2 will restart if exceeded - leaves room for OS)
- **5-minute timeout** for large video processing
- **30-second graceful timeout** for clean worker shutdown
- **5-second restart delay** to allow memory to be freed
- **Swap space required** (2GB recommended) - see T2_MICRO_SETUP.md

### For Larger Instances (2GB+ RAM):
To optimize for larger instances, update:
1. `max_memory_restart` in `ecosystem.config.js` to `1G` or `1.5G`
2. `--max-requests` in `run.sh` to `10-20`
3. `--worker-connections` in `run.sh` to `500-1000`

**Important:** If you're still experiencing OOM kills:
1. Check system memory: `free -h` and `cat /proc/meminfo`
2. Ensure swap space is configured (critical for t2.micro)
3. Reduce `max_memory_restart` in `ecosystem.config.js` further
4. Reduce `--max-requests` in `run.sh` to `3-5`
5. Monitor memory usage: `pm2 monit`

## Troubleshooting

### Check if service is running:
```bash
pm2 status
pm2 logs siha-ai
```

### Check memory usage:
```bash
pm2 monit
```

### View error logs:
```bash
tail -f ~/.pm2/logs/siha-ai-error.log
```

### Restart after code changes:
```bash
# Pull latest code
git pull

# Restart service
pm2 restart siha-ai
```

### If service keeps crashing (OOM kills):
1. Check system memory: `free -h` and `cat /proc/meminfo`
2. Check PM2 memory usage: `pm2 monit` (watch the memory column)
3. Check logs: `pm2 logs siha-ai --err`
4. Check system logs for OOM kills: `dmesg | grep -i "out of memory"` or `journalctl -k | grep -i oom`
5. Reduce `max_memory_restart` in `ecosystem.config.js` (currently 1.5G, try 1G)
6. Reduce `--max-requests` in `run.sh` (currently 20, try 15 or 10)
7. Check if other processes are using too much memory: `ps aux --sort=-%mem | head -20`

## Health Check

Test the service:
```bash
curl http://localhost:3001/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "ai-analysis-python",
  "modelsLoaded": true,
  "timestamp": "2025-11-21T16:30:00"
}
```

