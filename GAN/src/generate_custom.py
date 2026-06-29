"""
自定义随机数，用训练好的 GAN Generator 生成 8 张 FashionMNIST 图片。

用法：
  python generate_custom.py                  # 默认随机种子
  python generate_custom.py --seed 42        # 固定随机种子，每次生成相同图片
  python generate_custom.py --seed 123 --save  # 手动保存图片
"""

import argparse
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# 复用 GAN.py 中的 Generator 定义
from GAN import Generator


def generate_images(z_dim=100, n_images=8, seed=None, device=None):
    """
    生成 n_images 张图片。

    参数:
        seed: 如果为 None，每次生成不同图片；否则固定随机种子。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 加载训练好的 Generator ----
    G = Generator(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load('generator.pth', map_location=device, weights_only=True))
    G.eval()  # 推理模式

    # ---- 自定义随机噪声 ----
    if seed is not None:
        torch.manual_seed(seed)          # 固定 PyTorch 随机数
        np.random.seed(seed)             # 固定 NumPy 随机数
        print(f'使用固定随机种子: seed={seed}')
    else:
        print('使用随机种子（每次结果不同）')

    # 生成 8 个随机噪声向量 (8, z_dim)
    z = torch.randn(n_images, z_dim, device=device)

    # 可选：你也可以完全手写自己的噪声向量
    # z = torch.tensor([...])  # 自己指定数值

    # ---- 生成图片 ----
    with torch.no_grad():
        imgs = G(z)                     # (8, 1, 28, 28), 值域 [-1, 1]

    # ---- 显示 ----
    grid = vutils.make_grid(imgs.cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 2))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title(f'Generated images (seed={seed})' if seed is not None else 'Generated images (random)')
    plt.tight_layout()
    plt.show()

    return imgs


def save_images(imgs, filename='generated_custom.png'):
    """保存生成的图片到文件。"""
    grid = vutils.make_grid(imgs.cpu(), nrow=8, normalize=True, pad_value=0.3)
    vutils.save_image(imgs.cpu(), filename, nrow=8, normalize=True, pad_value=0.3)
    print(f'图片已保存到: {filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='用 GAN Generator 自定义生成图片')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（固定后每次生成相同结果）')
    parser.add_argument('--save', action='store_true',
                        help='保存生成的图片到文件')
    parser.add_argument('--n', type=int, default=8,
                        help='生成图片数量（默认 8）')
    args = parser.parse_args()

    imgs = generate_images(n_images=args.n, seed=args.seed)

    if args.save:
        save_images(imgs)

    # ---- 额外技巧：你也可以手写噪声 ----
    # 如果你想完全手动控制，可以在运行后取消下面的注释：
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # G = Generator(z_dim=100).to(device)
    # G.load_state_dict(torch.load('generator.pth', map_location=device, weights_only=True))
    # G.eval()
    #
    # # 手动构造 8 个噪声向量
    # my_noise = torch.tensor([
    #     [0.1] * 100,   # 全 0.1
    #     [0.5] * 100,   # 全 0.5
    #     [1.0] * 100,   # 全 1.0
    #     [-0.5] * 100,  # 全 -0.5
    #     [0.3] * 100,
    #     [0.7] * 100,
    #     [-0.2] * 100,
    #     [0.0] * 100,
    # ], dtype=torch.float32, device=device)
    #
    # with torch.no_grad():
    #     my_imgs = G(my_noise)
    # save_images(my_imgs, 'manual_noise.png')
