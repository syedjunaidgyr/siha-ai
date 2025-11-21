# AI Service Deployment Guide

> **For AWS EC2 deployment, see [EC2_DEPLOYMENT.md](./EC2_DEPLOYMENT.md) for EC2-specific configurations.**

## Quick Restart

After deploying code changes, restart the service:

```bash
# SSH to server
cd /srv/siha/ai-service-python  # or your actual path

# Restart with PM2
pm2 restart siha-ai

# Or if using ecosystem config
pm2 restart ecosystem.config.js

# Check status
pm2 status
pm2 logs siha-ai --lines 50
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

The service is configured with aggressive memory management to prevent OOM kills:
- **1 worker process** (reduces memory usage)
- **Auto-restart after 20 requests** (prevents memory leaks - more frequent restarts)
- **1.5GB memory limit** (PM2 will restart if exceeded - reduced to prevent OOM kills before system limit)
- **5-minute timeout** for large video processing
- **30-second graceful timeout** for clean worker shutdown
- **4-second restart delay** to allow memory to be freed

**Important:** If you're still experiencing OOM kills, consider:
1. Reducing `max_memory_restart` in `ecosystem.config.js` to `1G` or `1.2G`
2. Reducing `--max-requests` in `run.sh` to `15` or `10`
3. Checking system memory: `free -h` and `cat /proc/meminfo`
4. Monitoring memory usage: `pm2 monit`

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

