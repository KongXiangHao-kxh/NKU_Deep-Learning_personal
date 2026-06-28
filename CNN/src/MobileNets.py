"""
使用 MobileNet (深度可分离卷积) 训练 CIFAR-10 图像分类器
======================================================
数据集: CIFAR-10 (10个类别: 飞机, 汽车, 鸟, 猫, 鹿, 狗, 青蛙, 马, 船, 卡车)
图像尺寸: 3×32×32 (RGB 三通道, 32×32 像素)

网络架构: 微型 MobileNet（适配 CIFAR-10 的 32×32 输入）
  - 核心组件: Depthwise Separable Convolution (深度可分离卷积)
    - Depthwise: 单通道 3×3 卷积 (groups=in_channels)
    - Pointwise: 1×1 卷积混合通道
    - 参数量 = 3×3×M + M×N, 远小于标准卷积的 3×3×M×N
  - 初始卷积: 3×3 (Keep 32×32 空间分辨率)
  - 多个 DSConv 模块堆叠
  - 全局平均池化 + 全连接层分类

参考文献: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications (Howard et al., 2017)
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
# MobileNet 核心组件: Depthwise Separable Convolution
# =============================================================================

class DepthwiseSepConv(nn.Module):
    """深度可分离卷积 (Depthwise Separable Convolution)

    MobileNet 的核心组件, 将标准卷积分解为两步:
      1. Depthwise Conv (3×3): 每个输入通道独立卷积 (groups=in_channels)
      2. Pointwise Conv (1×1): 1×1 卷积混合通道信息

    参数量对比 (以 in=M, out=N 为例):
      - 标准卷积: 3×3×M×N
      - 深度可分离卷积: 3×3×M + M×N = M×(9+N) ≈ M×N (当 N >> 9 时)
      - 压缩比 ≈ 1/N + 1/9, 通常可减少 8-9 倍参数量

    结构: Depthwise Conv3×3 → BN → ReLU → Pointwise Conv1×1 → BN → ReLU

    参数:
        in_planes:  输入通道数
        out_planes: 输出通道数
        stride:     卷积步长 (1 或 2)
    """

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        # ---- Depthwise: 3×3 卷积, groups=in_planes (逐通道) ----
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3,
                                    stride=stride, padding=1, groups=in_planes,
                                    bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)

        # ---- Pointwise: 1×1 卷积混合通道 ----
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class MobileNet(nn.Module):
    """MobileNet (适配 CIFAR-10 的 32×32 输入)

    与标准 MobileNet 的区别:
      - 初始卷积为 3×3 (非 7×7), 无 MaxPool, stride=1 (保留 32×32 分辨率)
      - 缩减的层数 (CIFAR 图像更小)
      - 通过 width_mult 参数控制网络宽度

    参数:
        width_mult:  宽度乘子 α (控制通道数, <1 则网络更窄, 默认 1.0)
        num_classes: 分类数 (CIFAR-10 = 10)
    """

    def __init__(self, width_mult=1.0, num_classes=10):
        super().__init__()
        self.width_mult = width_mult

        def _adjust_channels(ch):
            """应用宽度乘子, 确保至少为 8"""
            return max(8, int(ch * width_mult))

        # ---- 初始卷积: 3×3, stride=1 (保持 32×32) ----
        c1 = _adjust_channels(32)
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        # ---- 深度可分离卷积层堆叠 ----
        # 每层: (in_ch, out_ch, stride)
        # 空间尺寸变化: 32 → 32 → 16 → 16 → 8 → 8 → 4 → 4 → 4 → 4 → 2 → 2
        cfg = [
            (32, 64,  1),     # 32×32
            (64, 128, 2),     # 16×16
            (128, 128, 1),    # 16×16
            (128, 256, 2),    # 8×8
            (256, 256, 1),    # 8×8
            (256, 512, 2),    # 4×4
            (512, 512, 1),    # 4×4
            (512, 512, 1),    # 4×4
            (512, 512, 1),    # 4×4
            (512, 512, 1),    # 4×4
            (512, 1024, 2),   # 2×2
            (1024, 1024, 1),  # 2×2
        ]

        layers = []
        for in_ch, out_ch, stride in cfg:
            layers.append(DepthwiseSepConv(
                _adjust_channels(in_ch),
                _adjust_channels(out_ch),
                stride
            ))
        self.layers = nn.Sequential(*layers)

        # ---- 全局平均池化 + 全连接分类 ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(_adjust_channels(1024), num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =============================================================================
# 工厂函数: 不同规模的 MobileNet
# =============================================================================

def MobileNetV1(width_mult=1.0, num_classes=10):
    """标准 MobileNetV1 (适配 CIFAR-10)

    参数:
        width_mult: 宽度乘子
            - 1.0: 标准宽度 (约 3.2M 参数)
            - 0.75: 75% 宽度 (约 1.9M 参数)
            - 0.5: 50% 宽度 (约 0.9M 参数)
            - 0.25: 25% 宽度 (约 0.3M 参数)
    """
    return MobileNet(width_mult=width_mult, num_classes=num_classes)


def MiniMobileNet(num_classes=10):
    """微型 MobileNet (适配 CIFAR-10, 快速实验用)

    使用更少的层和更窄的通道, 适合快速验证.
    参数量约 0.3M (width_mult=0.5).
    """
    return MobileNet(width_mult=0.5, num_classes=num_classes)


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

    # CIFAR-10 数据增强与归一化（与 ResNet/DenseNet 一致）
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

    batch_size = 128  # MobileNet 参数量小, 可使用较大 batch_size

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
    # 2. 定义 MobileNet 网络
    # =========================================================================

    # 使用微型版本 (训练更快)
    net = MiniMobileNet(num_classes=10)
    # 也可使用标准版本 (效果更好但训练慢):
    # net = MobileNetV1(width_mult=1.0, num_classes=10)

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

    # MobileNet 参数量少, 使用较小的 weight_decay
    optimizer = optim.SGD(net.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=4e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # =========================================================================
    # 4. 训练网络
    # =========================================================================

    num_epochs = 30  # 与 ResNet/DenseNet 保持一致

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
    plt.title('Training Loss Curve (MobileNet)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 子图2: 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, test_accuracies, 'm-o', markersize=6, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Curve (MobileNet)')
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.grid(True, linestyle='--', alpha=0.6)
    # 标注每个点的数值
    for ep, acc in zip(epoch_list, test_accuracies):
        plt.annotate(f'{acc:.1f}%', (ep, acc), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('results/mobilenet_training_curves.png', dpi=150)
    plt.show()
    print('Training curves saved to results/mobilenet_training_curves.png')

    # =========================================================================
    # 保存模型
    # =========================================================================

    PATH = './checkpoint/cifar_mobilenet.pth'
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
    net = MiniMobileNet(num_classes=10)
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
