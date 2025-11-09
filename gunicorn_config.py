# gunicorn_config.py
timeout = 300  # 5 minutes instead of default 30 seconds
workers = 1    # Use only 1 worker to save memory
worker_class = 'sync'
max_requests = 100
max_requests_jitter = 10