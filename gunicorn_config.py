workers = 1  # 减少工作进程数量以降低内存使用，根据服务器性能调整
bind = '0.0.0.0:8008'  # 绑定 IP 和端口
timeout = 120
limit_request_line = 8190  # 设置请求行大小限制为8190
preload_app = False  # 禁用预加载应用，避免多进程环境下的C扩展和线程安全问题
# 内存相关优化
buffer_response = True
keepalive = 5

# 日志配置
loglevel = 'info'  # 降低日志级别以减少无效日志
accesslog = '-'  # 访问日志输出到标准输出
errorlog = '-'  # 错误日志输出到标准输出
capture_output = False  # 禁用捕获标准输出和标准错误，减少日志量
enable_stdio_inheritance = False  # 禁用子进程继承标准文件描述符