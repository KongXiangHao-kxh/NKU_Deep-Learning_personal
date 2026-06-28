"""
使用密集连接网络 (DenseNet) 训练 CIFAR-10 图像分类器
=============================================
数据集: CIFAR-10 (10个类别: 飞机, 汽车, 鸟, 猫, 鹿, 狗, 青蛙, 马, 船, 卡车)
图像尺寸: 3×32×32 (RGB 三通道, 32×32 像素)

网络架构: DenseNet-BC (适配 CIFAR-10 的 32×32 输入)
  - 初始卷积: 3×3, 不使用 7×7（CIFAR 分辨率较低）
  - 3 个 DenseBlock, 每个包含多个 Bottleneck 层
  - DenseBlock 之间由 Transition 层连接 (BN→Conv 1×1→AvgPool)
  - 全局平均池化 + 全连接层分类
  - 特点: 每层接收之前所有层的输出作为输入, 缓解梯度消失

参考文献: Densely Connected Convolutional Networks (Huang et al., CVPR 2017)
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
# DenseNet 核心组件
# =============================================================================

class Bottleneck(nn.Module):
    """DenseNet-BC 瓶颈层

    DenseNet-BC 的每个瓶颈层包含:
      BN → ReLU → Conv 1×1 (降维至 4×growth_rate) → BN → ReLU → Conv 3×3 (输出 growth_rate 个特征图)

    其中 "BC" 代表 Bottleneck + Compression.
    相比原始 DenseBlock (BN→ReLU→Conv3×3), 通过 1×1 卷积降低计算量.

    参数:
        in_planes:  输入通道数
        growth_rate: 增长率 k, 每层输出的新特征图数量
    """
    def __init__(self, in_planes, growth_rate):
        super().__init__()
        # 1×1 瓶颈: 将通道数降至 4×growth_rate (减少 3×3 卷积的计算量)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1,
                               stride=1, bias=False)
        # 3×3 卷积: 输出 growth_rate 个新特征图
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # 拼接 (concatenate) 所有之前的特征图和新的特征图
        out = torch.cat([x, out], 1)
        return out


class Transition(nn.Module):
    """DenseNet 过渡层 (位于两个 DenseBlock 之间)

    功能: 降维 (压缩通道) + 降采样 (缩小特征图尺寸)

    结构: BN → ReLU → Conv 1×1 → AvgPool 2×2

    参数:
        in_planes:  输入通道数
        out_planes: 输出通道数 (= compression × in_planes)
    """
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                              stride=1, bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class DenseNet(nn.Module):
    """密集连接网络 DenseNet (适配 CIFAR-10 的 32×32 输入)

    结构:
      1. 初始卷积: Conv 3×3 (保持空间尺寸)
      2. DenseBlock(1) → Transition(1) → DenseBlock(2) → Transition(2) → DenseBlock(3)
      3. BN → ReLU → 全局平均池化 → FC 分类

    参数:
        growth_rate:    增长率 k, 控制每层新增特征图数量 (默认 12)
        block_config:   每个 DenseBlock 中包含的 Bottleneck 层数 (默认 [12, 12, 12])
        num_classes:    分类数 (CIFAR-10 = 10)
        compression:    压缩因子 θ (0 < θ ≤ 1), Transition 层将通道数压缩为 θ×原始 (默认 0.5)
        init_planes:    初始卷积后的特征图数量 (默认 24)
    """

    def __init__(self, growth_rate=12, block_config=(12, 12, 12),
                 num_classes=10, compression=0.5, init_planes=24):
        super().__init__()
        self.growth_rate = growth_rate

        # ---- 初始卷积: 3×3, 输出 init_planes 个特征图 ----
        self.conv1 = nn.Conv2d(3, init_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(init_planes)

        # ---- 构建 DenseBlocks + Transition 层 ----
        num_planes = init_planes
        self.features = nn.Sequential()

        for i, num_layers in enumerate(block_config):
            # 当前 DenseBlock
            block = self._make_dense_block(num_planes, num_layers)
            self.features.add_module(f'denseblock_{i + 1}', block)
            num_planes += num_layers * growth_rate

            # DenseBlock 之间的 Transition 层 (最后一个 DenseBlock 之后不加)
            if i != len(block_config) - 1:
                out_planes = int(num_planes * compression)  # 压缩通道数
                trans = Transition(num_planes, out_planes)
                self.features.add_module(f'transition_{i + 1}', trans)
                num_planes = out_planes

        # ---- 最终 BN + ReLU ----
        self.bn_final = nn.BatchNorm2d(num_planes)

        # ---- 全局平均池化 + 全连接分类 ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_planes, num_classes)

    def _make_dense_block(self, in_planes, num_layers):
        """构建一个 DenseBlock

        包含 num_layers 个 Bottleneck 层, 每层输出 growth_rate 个新特征图.
        所有层之间密集连接: 第 L 层的输入 = 之前所有 L-1 层输出的拼接.

        参数:
            in_planes:  当前层的输入通道数
            num_layers: 该 DenseBlock 中包含的 Bottleneck 层数
        """
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.features(x)
        x = F.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =============================================================================
# 工厂函数: 不同规模的 DenseNet
# =============================================================================

def DenseNetBC_100(num_classes=10):
    """DenseNet-BC (L=100, k=12) — 适用于 CIFAR-10 的经典配置

    参考原论文 Table 1: DenseNet-BC k=12, depth=100 (CIFAR-10)
    block_config = [16, 16, 16] → 总共 3 个 DenseBlock, 各含 16 层
    """
    return DenseNet(growth_rate=12, block_config=(16, 16, 16),
                    num_classes=num_classes, compression=0.5, init_planes=24)


def DenseNetBC_40(num_classes=10):
    """DenseNet-BC (L=40, k=12) — 轻量级版本, 训练更快"""
    return DenseNet(growth_rate=12, block_config=(6, 6, 6),
                    num_classes=num_classes, compression=0.5, init_planes=16)


def MiniDenseNet(num_classes=10):
    """更小的 DenseNet (适配 CIFAR-10, 快速实验用)

    DenseBlock 各含 4 层, 增长率 k=8
    """
    return DenseNet(growth_rate=8, block_config=(4, 4, 4),
                    num_classes=num_classes, compression=0.5, init_planes=16)


def imshow(img):
    """显示一张图片（反归一化还原到 [0,1] 范围）"""
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    # =========================================================================
    # 1. 加载并归一化 CIFAR-10
    # =========================================================================

    # CIFAR-10 数据增强与归一化（与 ResNet 一致）
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

    batch_size = 64  # DenseNet 参数量大, 适当减小 batch_size

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
    # 2. 定义 DenseNet 网络
    # =========================================================================

    # 使用轻量级版本 (训练更快)
    net = MiniDenseNet(num_classes=10)
    # 也可使用标准版本 (更深的网络, 效果更好但训练慢):
    # net = DenseNetBC_100(num_classes=10)
    # net = DenseNetBC_40(num_classes=10)

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

    # DenseNet 参数量通常比 ResNet 少, 可适当减小 weight_decay
    optimizer = optim.SGD(net.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # =========================================================================
    # 4. 训练网络
    # =========================================================================

    num_epochs = 30  # 与 ResNet 保持一致

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
    plt.title('Training Loss Curve (DenseNet)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 子图2: 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, test_accuracies, 'g-o', markersize=6, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Curve (DenseNet)')
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.grid(True, linestyle='--', alpha=0.6)
    # 标注每个点的数值
    for ep, acc in zip(epoch_list, test_accuracies):
        plt.annotate(f'{acc:.1f}%', (ep, acc), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('results/densenet_training_curves.png', dpi=150)
    plt.show()
    print('Training curves saved to results/densenet_training_curves.png')

    # =========================================================================
    # 保存模型
    # =========================================================================

    PATH = './checkpoint/cifar_densenet.pth'
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
    net = MiniDenseNet(num_classes=10)
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
