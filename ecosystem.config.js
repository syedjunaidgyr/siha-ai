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
    max_memory_restart: '1500M',
    min_uptime: '10s',
    max_restarts: 10,
    restart_delay: 4000,
    kill_timeout: 5000,
    env: {
      NODE_ENV: 'production',
      FLASK_ENV: 'production',
      PORT: 3001,
      PYTHONHASHSEED: '0',
      MALLOC_ARENA_MAX: '2'
    },
    error_file: '/home/ec2-user/.pm2/logs/siha-ai-error.log',
    out_file: '/home/ec2-user/.pm2/logs/siha-ai-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    time: true
  }]
};
