#!/bin/bash
# 从 VNC 终端一键启动 Jupyter Lab，确保 Open3D 可视化窗口能正常弹出到 VNC 桌面
#
# 使用方法：
#   1. 在 VNC 里的 terminal 中执行：
#      ./start_jupyter_vnc.sh
#   2. 复制输出里的 URL（带 token）
#   3. 在 VS Code 里打开 visual.ipynb，选择 kernel -> "Existing Jupyter Server" -> 粘贴 URL

conda activate opensf

export DISPLAY=:1
export LD_LIBRARY_PATH=/opt/conda/envs/opensf/lib:$LD_LIBRARY_PATH
export OPEN3D_ENABLE_WEBRTC=0

jupyter lab --no-browser --ip=0.0.0.0 --port=8888
