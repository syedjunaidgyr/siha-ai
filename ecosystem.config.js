module.exports = {
  apps: [{
    name: 'siha-ai-sh',
    script: './run.sh',
    interpreter: '/bin/bash',
    cwd: '/srv/siha/ai-service-python', // Update this path to match your server
    instances: 1,
    exec_mode: 'fork',
    autorestart: true,
    watch: false,
    // Reduced memory threshold to restart before OOM kill
    // System OOM killer typically triggers around 80-90% of available memory
    // Setting to 1.5G gives buffer before system kills the process
    max_memory_restart: '1500M',
    // Restart after processing a certain number of requests to prevent memory leaks
    // This works in conjunction with gunicorn's --max-requests
    min_uptime: '10s', // Minimum uptime before considering it a successful start
    max_restarts: 10, // Max restarts in 1 minute window
    restart_delay: 4000, // Wait 4 seconds before restarting
    env: {
      NODE_ENV: 'production',
      PORT: 3001,
      FLASK_ENV: 'production',
      // Set Python memory limits (if available)
      PYTHONHASHSEED: '0', // Reproducible hashing, slightly reduces memory
      MALLOC_ARENA_MAX: '2' // Limit memory arenas (Linux only, helps with memory fragmentation)
    },
    error_file: '/home/ec2-user/.pm2/logs/siha-ai-sh-error.log',
    out_file: '/home/ec2-user/.pm2/logs/siha-ai-sh-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    time: true,
    // Kill timeout - if process doesn't exit gracefully, force kill after this time
    kill_timeout: 5000
  }]
};

