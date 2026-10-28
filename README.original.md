# 图像分类项目 README

## 项目概述
本项目包含两个主要的图像分类任务：一是使用 Google Teachable Machine 模型对猫的种类进行分类，支持静态图像预测和实时摄像头画面预测；二是对手写数字图像（MNIST 数据集）进行识别和分类。

## 代码文件说明
- `main.py`：项目主入口，解析命令行参数并调用对应分类器进行预测或训练。
- `run_mnist.sh`：运行手写数字识别任务，可指定训练轮次。
- `run_gtm_static.sh`：运行 Google Teachable Machine 静态图像预测任务，可指定测试图像。
- `run_gtm_realtime.sh`：运行 Google Teachable Machine 实时摄像头画面预测任务。
- `MNISTClassifier.py`：定义 MNIST 手写数字图像分类神经网络模型，含训练和评估方法。
- `GoogleTM_Image_CatClassifier.py`：定义调用 Google Teachable Machine 模型对猫种类分类的类，支持静态和实时预测。

## 文件夹说明
- `Dataset`：存放项目所需数据集，其中 `Cat` 文件夹中存放猫的类别的数据集，`MNIST` 数据集中存放手写数字识别的数据集。
- `GoogleTMModel`：包含 Google Teachable Machine 模型文件 model.tflite 和标签文件 labels.txt。
- `MNISTModel`：保存 MNIST 模型参数 model.pkl 和优化器状态 optimizer.pkl。

## 环境要求
- Python 3.6.5
- 安装以下依赖库：
```plaintext
numpy==1.19.5
opencv_python==4.5.4.60
Pillow==11.1.0
tensorflow==2.0.0
torch==1.8.0+cu111
torchvision==0.9.0+cu111
tqdm==4.64.1
```
- 安装 `conda` 并创建名为 `CVCourse` 的虚拟环境。

## 运行步骤

### 1. 创建并激活 conda 环境
```bash
conda create -n CVCourse python=3.6.5
conda activate CVCourse
```

### 2. 安装依赖库
在激活的 `CVCourse` 环境中，使用以下命令安装所需的依赖库：
```bash
pip install numpy==1.19.5
pip install tensorflow==2.0.0
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111
pip install opencv_python==4.5.4.60 Pillow==11.1.0 tqdm==4.64.1
```

### 3. 运行不同任务

#### 手写数字识别任务
使用 `run_mnist.sh` 脚本运行手写数字识别任务，可通过 `-e` 参数指定训练轮次。若不指定，默认训练 10 轮。
```bash
# 使用默认训练轮次
./run_mnist.sh

# 指定训练轮次为 20
./run_mnist.sh -e 20
```

#### Google Teachable Machine 静态图像预测任务
使用 `run_gtm_static.sh` 脚本运行静态图像预测任务，可通过 `-i` 参数指定测试图像。若不指定，使用默认图像 test 集中的 `BritishShorthair01.jpg`。
```bash
# 使用默认测试图像（BritishShorthair01.jpg）
./run_gtm_static.sh

# 指定测试图像
./run_gtm_static.sh -i your_test_image.jpg
```

#### Google Teachable Machine 实时摄像头画面预测任务
使用 `run_gtm_realtime.sh` 脚本运行实时预测任务，无需额外参数。
```bash
./run_gtm_realtime.sh
```

### 4. 停用 conda 环境（可选）
在完成所有任务后，可使用以下命令停用 `CVCourse` 环境：
```bash
conda deactivate
```

## 注意事项
- 确保 `GoogleTMModel` 目录下包含 `model.tflite` 和 `labels.txt` 文件，以及 `Dataset` 目录结构正确。
- 在运行实时预测任务时，确保摄像头设备正常工作。 