#!/bin/bash
# Google Teachable Machine 对摄像头图像进行处理
export LANG=zh_CN.UTF-8

# 获取 conda 初始化命令
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate CVCourse

# 检查 conda 环境是否激活成功
if [ $? -eq 0 ]; then
    echo "成功激活 CVCourse conda 环境。"
    # 运行 main.py 文件
    python main.py --gtm --realtime
    # 检查脚本是否成功运行
    if [ $? -eq 0 ]; then
        echo "main.py 脚本成功运行。"
    else
        echo "main.py 脚本运行失败。"
    fi
else
    echo "无法激活 CVCourse conda 环境。"
fi

# 可选：停用 conda 环境
conda deactivate

# 等待用户输入，按任意键后窗口才会关闭
read -p "脚本运行结束，按任意键关闭窗口..." -n 1 -s
