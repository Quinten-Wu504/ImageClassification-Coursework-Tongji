import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class MNISTClassifier(nn.Module):
    """
    定义了一个用于对 MNIST 手写数字图像进行分类的神经网络模型
    """
    # 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        # 继承 nn.Module 的构造函数
        super(MNISTClassifier, self).__init__()
        # 定义训练时和测试时的批次大小
        self.train_batch_size = 128
        self.test_batch_size = 1000
        # 定义第一个全连接层，输入维度是图像的像素数（1 * 28 * 28），输出维度是 100
        self.fc1 = nn.Linear(1 * 28 * 28, 100)
        # 定义第二个全连接层，输入维度是 100，输出维度是 10（对应 0-9 的 10 个数字类别）
        self.fc2 = nn.Linear(100, 10)
        # 定义 Adam 优化器，学习率设置为 1e-3
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, image):
        # 将输入图像张量展平成一维向量
        image = image.view(-1, 1 * 28 * 28)
        # 将展开后的图像向量输入到第一个全连接层
        output_result = self.fc1(image)
        # 对第一个全连接层的输出应用 ReLU 激活函数
        output_result = F.relu(output_result)
        # 将经过 ReLU 激活后的结果输入到第二个全连接层
        output_result = self.fc2(output_result)
        # 对第二个全连接层的输出应用对数softmax函数，得到每个类别的对数概率
        output_result = F.log_softmax(output_result, dim=-1)
        return output_result

    def get_mnist_dataloader(self, train=True):
        # 定义图像预处理的操作，包括将图像转换为张量和归一化处理
        func = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.1307,),
                std=(0.3081,)
            )
        ])

        # 加载MNIST数据集，根据train参数决定是加载训练集还是测试集
        mnist = MNIST(root="./Dataset", train=train, download=True, transform=func)
        # 根据train参数选择批次大小
        batch_size = self.train_batch_size if train else self.test_batch_size
        # 创建数据加载器，用于批量加载数据
        return DataLoader(mnist, batch_size=batch_size, shuffle=True)

    def mnist_train(self, train_epoch):
        # 获取训练数据加载器
        train_dataloader = self.get_mnist_dataloader(train=True)
        # 使用tqdm创建一个进度条，方便查看训练进度
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        # 用于存储每个批次的损失值
        total_loss = []
        for idx, (input_feature, target_feature) in bar:
            # 将输入特征和目标特征移动到指定的设备（GPU或CPU）上
            input_feature = input_feature.to(self.device)
            target_feature = target_feature.to(self.device)
            # 清零优化器中的梯度信息
            self.optimizer.zero_grad()
            # 前向传播，得到模型的输出
            output_result = self(input_feature)
            # 计算负对数似然损失
            loss = F.nll_loss(output_result, target_feature)
            # 反向传播，计算梯度
            loss.backward()
            # 将当前批次的损失值添加到总损失列表中
            total_loss.append(loss.item())
            # 更新模型的参数
            self.optimizer.step()
            # 打印训练情况
            if idx % 10 == 0:
                bar.set_description("epoch:{} idx:{},loss:{:.6f}".format(train_epoch, idx, np.mean(total_loss)))
                try:
                    torch.save(self.state_dict(), "./MNISTModel/model.pkl")
                    torch.save(self.optimizer.state_dict(), "./MNISTModel/optimizer.pkl")
                except Exception as e:
                    print(f"Error saving model and optimizer: {e}")

    def mnist_eval(self):
        # 检查是否存在保存的模型参数文件
        if os.path.exists("./models/model.pkl"):
            # 加载保存的模型参数
            self.load_state_dict(torch.load("./models/model.pkl"))

        # 获取测试数据加载器
        test_dataloader = self.get_mnist_dataloader(train=False)

        # 初始化用于存储每个批次的损失值/准确率的列表
        total_loss = []
        total_acc = []

        # 关闭梯度计算，减少内存消耗，提高推理速度
        with torch.no_grad():
            for input_feature, target_feature in test_dataloader:
                # 将输入特征和目标特征移动到指定的设备（GPU或CPU）上
                input_feature = input_feature.to(self.device)
                target_feature = target_feature.to(self.device)
                # 前向传播，得到模型的输出
                output_feature = self(input_feature)
                # 计算负对数似然损失
                loss = F.nll_loss(output_feature, target_feature)
                # 将当前批次的损失值添加到总损失列表中
                total_loss.append(loss.item())
                # 获取预测的类别索引
                predictions = output_feature.max(dim=-1)[-1]
                # 计算当前批次的准确率，并添加到总准确率列表中
                total_acc.append(predictions.eq(target_feature).float().mean().item())

        # 打印测试集的平均损失和平均准确率
        print("test loss:{},test acc:{}".format(np.mean(total_loss), np.mean(total_acc)))

        return np.mean(total_loss), np.mean(total_acc)


if __name__ == "__main__":
    model = MNISTClassifier().to(MNISTClassifier.device)
    epochs = 10
    for epoch in range(epochs):
        model.mnist_train(epoch)
        model.mnist_eval()
