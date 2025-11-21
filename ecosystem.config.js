module.exports = {
  apps: [{
    name: 'siha-ai',
    script: './run.sh',
    interpreter: '/bin/bash',
    cwd: '/srv/siha/ai-service-python',
    instances: 1,
    exec_mode: 'fork',
    autorestart: true,
    watch: false,
    // For t2.micro (1GB RAM): Use 400MB limit to leave room for OS and other processes
    // For larger instances, increase to 1G or 1.5G
    max_memory_restart: '400M',
    min_uptime: '10s',
    max_restarts: 15, // More restarts allowed for memory-constrained environments
    restart_delay: 5000, // Longer delay to ensure memory is freed
    kill_timeout: 5000,
    env: {
      NODE_ENV: 'production',
      FLASK_ENV: 'production',
      PORT: 3001,
      PYTHONHASHSEED: '0',
      MALLOC_ARENA_MAX: '2', // Limit memory arenas to reduce fragmentation
      PYTHONUNBUFFERED: '1', // Disable buffering for better memory usage
      // Limit Python memory growth (if using memory_profiler or similar)
      MALLOC_TRIM_THRESHOLD_: '131072' // 128KB - more aggressive memory trimming
    },
    error_file: '/home/ec2-user/.pm2/logs/siha-ai-error.log',
    out_file: '/home/ec2-user/.pm2/logs/siha-ai-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    time: true
  }]
};
