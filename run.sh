#!/bin/bash

# 设置环境变量以支持WebSocket
export PYTHONUNBUFFERED=1

# 使用gunicorn启动应用，启用WebSocket支持
gunicorn --config gunicorn_config.py app:app