"""Logs a simple transform hierarchy."""

import rerun as rr
import time

rr.init("rerun_example_transform3d_hierarchy_simple")

rr.serve(web_port=9090)

# 想可视化什么
# 1. 一个点云帧里的所有动态簇与实际的bbox，bbox上显示实例ID与实例类别
# 2. 

print("Web Viewer 正在运行，请在浏览器中打开链接！")
print("按 Ctrl+C 退出程序...")

try:
    while True:
        time.sleep(1)  # 保持 Python 进程不退出
except KeyboardInterrupt:
    print("程序已退出")