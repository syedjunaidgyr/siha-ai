# AWS EC2 Deployment Guide for AI Service

This guide covers EC2-specific configurations and optimizations for the Python AI Service.

## Prerequisites

1. **EC2 Instance Requirements:**
   - **Minimum**: t3.medium (2 vCPU, 4GB RAM) - for light usage
   - **Recommended**: t3.large (2 vCPU, 8GB RAM) or t3.xlarge (4 vCPU, 16GB RAM) - for production
   - **For heavy ML workloads**: Consider c5.xlarge or c5.2xlarge (compute optimized)

2. **Operating System**: Amazon Linux 2 or Ubuntu 20.04/22.04 LTS

## Step 1: Security Group Configuration

Ensure your EC2 security group allows inbound traffic on port 3001:

```bash
# Via AWS Console:
# 1. Go to EC2 → Security Groups
# 2. Select your instance's security group
# 3. Add inbound rule:
#    - Type: Custom TCP
#    - Port: 3001
#    - Source: Your backend's security group (or specific IP)
#    - Description: AI Service

# Or via AWS CLI:
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 3001 \
  --source-group sg-yyyyyyyyy  # Backend security group
```

**Important**: For security, only allow access from your backend service, not from the internet (0.0.0.0/0).

## Step 2: System Limits Configuration

EC2 instances may have default limits that need adjustment for production workloads:

```bash
# Check current limits
ulimit -a

# Edit limits configuration
sudo nano /etc/security/limits.conf

# Add these lines (adjust values based on your instance size):
* soft nofile 65535
* hard nofile 65535
* soft nproc 32768
* hard nproc 32768
* soft memlock unlimited
* hard memlock unlimited

# For systemd services (PM2), also edit:
sudo nano /etc/systemd/system.conf

# Uncomment and set:
DefaultLimitNOFILE=65535
DefaultLimitNPROC=32768

# Reload systemd
sudo systemctl daemon-reload
```

## Step 3: Memory and Swap Configuration

For instances with limited RAM, configure swap space:

```bash
# Check current swap
free -h
swapon --show

# Create swap file (if needed, adjust size based on instance)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make it permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Optimize swap usage (reduce swappiness)
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**Note**: Swap is slower than RAM. If you're using swap frequently, consider upgrading your instance type.

## Step 4: Install Dependencies

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@your-ec2-ip

# Navigate to service directory
cd /srv/siha/ai-service-python

# Install Python 3.11+ (if not already installed)
# Amazon Linux 2:
sudo amazon-linux-extras install python3.11
# Ubuntu:
sudo apt update && sudo apt install python3.11 python3.11-venv

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 5: Configure PM2 for Auto-Start

```bash
# Install PM2 globally (if not already installed)
npm install -g pm2

# Start the service
cd /srv/siha/ai-service-python
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Setup PM2 to start on system boot
pm2 startup
# Follow the instructions shown (will be something like):
# sudo env PATH=$PATH:/usr/bin pm2 startup systemd -u ec2-user --hp /home/ec2-user
```

## Step 6: Verify Service is Running

```bash
# Check PM2 status
pm2 status

# Check if service is listening on port 3001
sudo netstat -tlnp | grep 3001
# or
sudo ss -tlnp | grep 3001

# Test health endpoint
curl http://localhost:3001/health

# Check logs
pm2 logs siha-ai --lines 50
```

## Step 7: EC2-Specific Optimizations

### A. Instance Metadata Service (IMDS) - For AWS Credentials

If your service needs AWS credentials (for S3 access), use IAM roles instead of hardcoded keys:

```bash
# Attach IAM role to EC2 instance with S3 permissions
# No code changes needed - boto3 will automatically use instance role
```

### B. CloudWatch Logs Integration

Send PM2 logs to CloudWatch:

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
sudo rpm -U ./amazon-cloudwatch-agent.rpm

# Configure (optional, PM2 logs are already in ~/.pm2/logs/)
```

### C. Memory Monitoring

Set up CloudWatch alarms for memory usage:

```bash
# Via AWS Console:
# 1. CloudWatch → Alarms → Create Alarm
# 2. Metric: EC2 → Per-Instance Metrics → MemoryUtilization
# 3. Threshold: > 80% for 5 minutes
# 4. Action: SNS notification or auto-scaling
```

## Step 8: Network Configuration

Ensure the service binds to all interfaces (0.0.0.0) for internal EC2 communication:

```bash
# Verify in run.sh that gunicorn uses:
# -b 0.0.0.0:3001  (not 127.0.0.1)

# Test connectivity from backend
# On backend EC2 instance:
curl http://<ai-service-private-ip>:3001/health
```

## Troubleshooting

### Service Won't Start

```bash
# Check PM2 logs
pm2 logs siha-ai --err

# Check system logs
sudo journalctl -u pm2-ec2-user -n 50

# Check if port is in use
sudo lsof -i :3001

# Check Python/gunicorn
cd /srv/siha/ai-service-python
source venv/bin/activate
python -c "import flask; print('OK')"
gunicorn --version
```

### Out of Memory (OOM) Kills

```bash
# Check system memory
free -h
cat /proc/meminfo

# Check OOM kills
dmesg | grep -i "out of memory"
journalctl -k | grep -i oom

# Check PM2 memory usage
pm2 monit

# Solutions:
# 1. Reduce max_memory_restart in ecosystem.config.js (currently 1500M)
# 2. Reduce --max-requests in run.sh (currently 20)
# 3. Upgrade EC2 instance type
# 4. Add swap space (temporary solution)
```

### Connection Refused

```bash
# Check if service is running
pm2 status

# Check if port is listening
sudo netstat -tlnp | grep 3001

# Check security group rules
# Ensure port 3001 is open in EC2 security group

# Check firewall (if enabled)
sudo iptables -L -n
sudo ufw status  # Ubuntu
```

### High Memory Usage

```bash
# Monitor in real-time
pm2 monit

# Check process memory
ps aux --sort=-%mem | head -20

# Check Python memory usage
# Install memory_profiler if needed
pip install memory-profiler

# Restart service to free memory
pm2 restart siha-ai
```

## Performance Tuning

### For t3.medium (4GB RAM):
- Keep `max_memory_restart: '1G'` in ecosystem.config.js
- Reduce `--max-requests` to `15` in run.sh
- Consider using swap space

### For t3.large (8GB RAM):
- Current settings should work: `max_memory_restart: '1.5G'`
- Can increase `--max-requests` to `25` if stable

### For t3.xlarge+ (16GB+ RAM):
- Can increase `max_memory_restart: '2G'`
- Can increase `--max-requests` to `30-50`
- Consider using `-w 2` (2 workers) if CPU-bound

## Backup and Recovery

```bash
# Backup PM2 configuration
pm2 save
cp ~/.pm2/dump.pm2 ~/.pm2/dump.pm2.backup

# Backup service code
cd /srv/siha
tar -czf ai-service-backup-$(date +%Y%m%d).tar.gz ai-service-python/

# Restore PM2 processes
pm2 resurrect
```

## Monitoring Commands

```bash
# Real-time monitoring
pm2 monit

# Check service status
pm2 status
pm2 info siha-ai

# View logs
pm2 logs siha-ai
pm2 logs siha-ai --lines 100
pm2 logs siha-ai --err

# Check system resources
htop  # or top
free -h
df -h
```

## Security Best Practices

1. **Don't expose port 3001 to the internet** - only allow from backend security group
2. **Use IAM roles** instead of hardcoded AWS credentials
3. **Keep dependencies updated**: `pip list --outdated`
4. **Use HTTPS** if exposing externally (via ALB/NLB)
5. **Regular backups** of configuration and code
6. **Monitor logs** for suspicious activity

## Cost Optimization

1. **Use Spot Instances** for non-critical workloads (with proper fallback)
2. **Right-size instances** - monitor CPU/memory usage and adjust
3. **Use Reserved Instances** for predictable workloads
4. **Enable CloudWatch detailed monitoring** only if needed (costs extra)
5. **Clean up old logs**: `pm2 flush` (after backing up if needed)

