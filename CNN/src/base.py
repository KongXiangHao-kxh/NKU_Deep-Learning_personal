"""
使用卷积神经网络 (CNN) 训练 CIFAR-10 图像分类器
==============================================
数据集: CIFAR-10 (10个类别: 飞机, 汽车, 鸟, 猫, 鹿, 狗, 青蛙, 马, 船, 卡车)
图像尺寸: 3×32×32 (RGB 三通道, 32×32 像素)

步骤:
  1. 加载并归一化 CIFAR-10 数据集
  2. 定义卷积神经网络
  3. 定义损失函数和优化器
  4. 训练网络
  5. 测试网络
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    """显示一张图片（反归一化还原到 [0,1] 范围）"""
    img = img / 2 + 0.5        # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    """一个简单的 CNN, 2层卷积 + 3层全连接

    网络结构:
        Input (3×32×32)
            │
        Conv2d(3→6, 5×5) + ReLU + MaxPool(2×2)    → (6×14×14)
            │
        Conv2d(6→16, 5×5) + ReLU + MaxPool(2×2)   → (16×5×5)
            │
        Flatten                                    → (400)
            │
        Linear(400→120) + ReLU                     → (120)
        Linear(120→84)  + ReLU                     → (84)
        Linear(84→10)                              → (10)
    """

    def __init__(self):
        super().__init__()
        # 卷积层1: 3通道 → 6通道, 5×5卷积核, 输出特征图 28×28, 池化后 14×14
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2: 6通道 → 16通道, 5×5卷积核, 输出特征图 10×10, 池化后 5×5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层: 展平后 16×5×5=400 → 120 → 84 → 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积块1: 卷积 → ReLU → 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积块2: 卷积 → ReLU → 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平 (保留 batch 维度)
        x = torch.flatten(x, 1)
        # 全连接分类器
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)          # 输出 10 个原始分数 (logits)
        return x


def main():
    """主流程: 加载数据 → 定义网络 → 训练 → 测试"""
    # =========================================================================
    # 1. 加载并归一化 CIFAR-10
    # =========================================================================

    # 将 PILImage ([0, 1]) 转换为 Tensor ([-1, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4

    # 训练集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 展示一些随机训练图片
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # =========================================================================
    # 2. 定义卷积神经网络
    # =========================================================================

    net = Net()
    print(net)

    # =========================================================================
    # 3. 定义损失函数和优化器
    # =========================================================================

    criterion = nn.CrossEntropyLoss()           # 交叉熵损失 (包含 Softmax)
    optimizer = optim.SGD(net.parameters(),     # 随机梯度下降 + 动量
                          lr=0.001, momentum=0.9)

    # =========================================================================
    # 4. 训练网络
    # =========================================================================

    # 记录训练过程中的 loss 和每个 epoch 后的测试准确率
    train_losses = []          # 每 2000 个 batch 记录一次平均 loss
    train_steps = []           # 对应的 step 编号
    test_accuracies = []       # 每个 epoch 结束后的测试准确率
    epoch_list = []            # 对应的 epoch 编号
    global_step = 0            # 全局 step 计数器（用于横轴）

    for epoch in range(10):                      # 遍历整个数据集 10 次
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data               # 获取一个 batch 的输入和标签

            optimizer.zero_grad()               # 清零梯度（避免累积）

            outputs = net(inputs)               # 前向传播: 输入 → 预测
            loss = criterion(outputs, labels)   # 计算损失
            loss.backward()                     # 反向传播: 计算梯度
            optimizer.step()                    # 更新参数

            # 打印统计并记录 loss
            running_loss += loss.item()
            global_step += 1
            if i % 2000 == 1999:                # 每 2000 个 mini-batch 打印一次
                avg_loss = running_loss / 2000
                print(f'[Epoch {epoch + 1}, Step {i + 1:5d}] loss: {avg_loss:.3f}')
                train_losses.append(avg_loss)
                train_steps.append(global_step)
                running_loss = 0.0

        # ---- 每个 epoch 结束后在测试集上评估准确率 ----
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        test_accuracies.append(acc)
        epoch_list.append(epoch + 1)
        print(f'===== Epoch {epoch + 1} test accuracy: {acc:.2f}% =====')

    print('Finished Training')

    # =========================================================================
    # 绘制训练 loss 曲线和测试准确率曲线
    # =========================================================================
    plt.figure(figsize=(12, 5))

    # 子图1: Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_losses, 'b-o', markersize=4, linewidth=1.5)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 子图2: 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, test_accuracies, 'r-s', markersize=8, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Curve')
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.grid(True, linestyle='--', alpha=0.6)
    # 标注每个点的数值
    for ep, acc in zip(epoch_list, test_accuracies):
        plt.annotate(f'{acc:.1f}%', (ep, acc), textcoords='offset points',
                     xytext=(0, 12), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150)
    plt.show()
    print('Training curves saved to results/training_curves.png')

    # =========================================================================
    # 保存模型
    # =========================================================================

    PATH = './checkpoint/cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')

    # =========================================================================
    # 5. 测试网络
    # =========================================================================

    # ---- 5a. 展示测试集图片 ----
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # ---- 5b. 加载模型并预测 ----
    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))
    net.eval()                                  # 切换到评估模式

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

    # ---- 5c. 评估整体准确率 ----
    correct = 0
    total = 0
    with torch.no_grad():                       # 测试阶段不需要计算梯度
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: '
          f'{100 * correct // total} %')

    # ---- 5d. 按类别评估准确率 ----
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print('\nPer-class accuracy:')
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    main()
