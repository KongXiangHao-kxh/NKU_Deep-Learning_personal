"""
使用残差网络 (ResNet) 训练 CIFAR-10 图像分类器
=============================================
数据集: CIFAR-10 (10个类别: 飞机, 汽车, 鸟, 猫, 鹿, 狗, 青蛙, 马, 船, 卡车)
图像尺寸: 3×32×32 (RGB 三通道, 32×32 像素)

网络架构: 微型 ResNet（适配 CIFAR-10 的 32×32 输入）
  - 初始卷积: 3×3, 不使用 7×7（CIFAR 分辨率较低）
  - 4 个残差阶段, 通道数: 16 → 32 → 64 → 128
  - 每个阶段 2 个 BasicBlock（含 BatchNorm）
  - 全局平均池化 + 全连接层分类
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# ResNet 基础组件: BasicBlock
# =============================================================================

class BasicBlock(nn.Module):
    """ResNet 基本残差块

    结构: Conv3×3 → BN → ReLU → Conv3×3 → BN → (+ skip) → ReLU

    当 stride≠1 或 in_planes≠planes 时, skip connection 通过 1×1 卷积调整维度.
    """
    expansion = 1  # 输出通道数相对于 planes 的倍数

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # 主路径: 两个 3×3 卷积
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 捷径 (shortcut): 若尺寸或通道数不匹配, 用 1×1 卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # 残差连接
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """微型 ResNet (适配 CIFAR-10 的 32×32 输入)

    与标准 ResNet 的区别:
      - 初始卷积为 3×3 (非 7×7), 无 MaxPool (保留更多空间信息)
      - 通道数从 16 开始 (非 64), 更适合小尺寸图像

    参数:
        block:       残差块类型 (BasicBlock)
        num_blocks:  每个阶段的残差块数量列表
        num_classes: 分类数 (CIFAR-10 = 10)
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        # 初始卷积: 3×3, 无 MaxPool
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 4 个残差阶段, 通道数逐步翻倍: 16 → 32 → 64 → 128
        self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)

        # 全局平均池化 + 全连接分类
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """构建一个残差阶段"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18(num_classes=10):
    """构建 ResNet-18 (适用于 CIFAR-10)"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=10):
    """构建 ResNet-34 (适用于 CIFAR-10)"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def MiniResNet(num_classes=10):
    """构建更小的 ResNet (通道更少, 适配 CIFAR-10)

    每个阶段只保留 2 个 BasicBlock, 通道数从 16 开始.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def imshow(img):
    """显示一张图片（反归一化还原到 [0,1] 范围）"""
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    """主流程: 加载数据 → 定义网络 → 训练 → 测试"""
    # =========================================================================
    # 1. 加载并归一化 CIFAR-10
    # =========================================================================

    # CIFAR-10 数据增强与归一化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),     # 随机裁剪 (数据增强)
        transforms.RandomHorizontalFlip(),        # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    batch_size = 128   # ResNet 使用更大的 batch size

    # 训练集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 展示一些随机训练图片
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))

    # =========================================================================
    # 2. 定义 ResNet 网络
    # =========================================================================

    net = MiniResNet(num_classes=10)
    print(net)

    # 计算参数量
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    # =========================================================================
    # 3. 定义损失函数和优化器
    # =========================================================================

    criterion = nn.CrossEntropyLoss()

    # 使用带动量的 SGD + 学习率衰减 (余弦退火)
    optimizer = optim.SGD(net.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # =========================================================================
    # 4. 训练网络
    # =========================================================================

    num_epochs = 30  # 适当增加 epoch 以发挥 ResNet 能力

    train_losses = []          # 每 100 个 batch 记录一次平均 loss
    train_steps = []           # 对应的 step 编号
    test_accuracies = []       # 每个 epoch 结束后的测试准确率
    epoch_list = []            # 对应的 epoch 编号
    global_step = 0            # 全局 step 计数器

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1
            if i % 100 == 99:   # 每 100 个 batch 记录一次
                avg_loss = running_loss / 100
                train_losses.append(avg_loss)
                train_steps.append(global_step)
                running_loss = 0.0

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ---- 每个 epoch 结束后在测试集上评估准确率 ----
        net.eval()
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

        print(f'[Epoch {epoch + 1:3d}/{num_epochs}] '
              f'Loss: {train_losses[-1]:.4f} | '
              f'Acc: {acc:.2f}% | '
              f'LR: {current_lr:.4f}')

    print('Finished Training')

    # =========================================================================
    # 绘制训练 loss 曲线和测试准确率曲线
    # =========================================================================
    plt.figure(figsize=(14, 5))

    # 子图1: Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_losses, 'b-', linewidth=1.0, alpha=0.7)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (ResNet)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 子图2: 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, test_accuracies, 'r-o', markersize=6, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Curve (ResNet)')
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.grid(True, linestyle='--', alpha=0.6)
    # 标注每个点的数值
    for ep, acc in zip(epoch_list, test_accuracies):
        plt.annotate(f'{acc:.1f}%', (ep, acc), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('results/resnet_training_curves.png', dpi=150)
    plt.show()
    print('Training curves saved to results/resnet_training_curves.png')

    # =========================================================================
    # 保存模型
    # =========================================================================

    PATH = './checkpoint/cifar_resnet.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')

    # =========================================================================
    # 5. 测试网络
    # =========================================================================

    # ---- 5a. 展示测试集图片 ----
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))

    # ---- 5b. 加载模型并预测 ----
    net = MiniResNet(num_classes=10)
    net.load_state_dict(torch.load(PATH, weights_only=True))
    net.eval()

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted:    ', ' '.join(f'{classes[predicted[j]]:5s}'
                                     for j in range(8)))

    # ---- 5c. 评估整体准确率 ----
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\nAccuracy of the network on the 10000 test images: '
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
