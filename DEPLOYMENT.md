# AI Service Deployment Guide

## Quick Restart

After deploying code changes, restart the service:

```bash
# SSH to server
cd /srv/siha/ai-service-python  # or your actual path

# Restart with PM2
pm2 restart siha-ai-sh

# Or if using ecosystem config
pm2 restart ecosystem.config.js

# Check status
pm2 status
pm2 logs siha-ai-sh --lines 50
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

The service is configured with:
- **1 worker process** (reduces memory usage)
- **Auto-restart after 30 requests** (prevents memory leaks - reduced from 50)
- **2GB memory limit** (PM2 will restart if exceeded - increased for large payloads and ML models)
- **5-minute timeout** for large video processing

## Troubleshooting

### Check if service is running:
```bash
pm2 status
pm2 logs siha-ai-sh
```

### Check memory usage:
```bash
pm2 monit
```

### View error logs:
```bash
tail -f ~/.pm2/logs/siha-ai-sh-error.log
```

### Restart after code changes:
```bash
# Pull latest code
git pull

# Restart service
pm2 restart siha-ai-sh
```

### If service keeps crashing:
1. Check memory: `free -h`
2. Check logs: `pm2 logs siha-ai-sh --err`
3. Reduce worker count in `run.sh` (already set to 1)
4. Reduce `--max-requests` in `run.sh` (currently 30)

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

