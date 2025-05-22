import sys
import os
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设脚本在 tools/ 目录下）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)
