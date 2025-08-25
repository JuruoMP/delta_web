workers = 1  # 可以根据服务器性能调整工作进程数
bind = '0.0.0.0:8008'  # 绑定 IP 和端口
timeout = 120
limit_request_line = 16382  # 设置请求行大小限制为8190