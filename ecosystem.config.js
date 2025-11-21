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
    max_memory_restart: '1G', // Restart if memory exceeds 1GB
    env: {
      NODE_ENV: 'production',
      PORT: 3001,
      FLASK_ENV: 'production'
    },
    error_file: '/home/ec2-user/.pm2/logs/siha-ai-sh-error.log',
    out_file: '/home/ec2-user/.pm2/logs/siha-ai-sh-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    time: true
  }]
};

