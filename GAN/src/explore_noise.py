
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

from GAN import Generator


def load_generator(pth_path='generator.pth', z_dim=100, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    G.eval()
    return G, device


def make_noise_batch(n_images, z_dim, base_seed=0, device='cpu'):
    """用固定种子生成一批 base noise，保证可复现。"""
    torch.manual_seed(base_seed)
    return torch.randn(n_images, z_dim, device=device)  # (8, 100)


def vary_one_dim(noise, dim_idx, values):
    """
    对 noise 的 dim_idx 列分别设置为 values 中的各个值。
    返回列表，每个元素是 (value, modified_noise_tensor)。
    """
    results = []
    for val in values:
        modified = noise.clone()
        modified[:, dim_idx] = val
        results.append((val, modified))
    return results


def generate_images(G, noise_batch, device):
    """生成图片并返回 (8, 1, 28, 28) 的 tensor。"""
    with torch.no_grad():
        imgs = G(noise_batch.to(device)).cpu()
    return imgs


def plot_experiment(all_results, chosen_dims, dim_values, save_dir='.'):
    """
    绘制实验结果：每个维度单独出一张图。
    all_results[dim_idx][val_idx] = generated images (8, 1, 28, 28)
    """
    n_vals = len(dim_values)

    for d_idx, dim in enumerate(chosen_dims):
        fig, axes = plt.subplots(n_vals, 1, figsize=(8, n_vals * 2.5))

        for v_idx in range(n_vals):
            ax = axes[v_idx] if n_vals > 1 else axes
            imgs = all_results[d_idx][v_idx]  # (8, 1, 28, 28)

            grid = vutils.make_grid(imgs, nrow=8, normalize=True, pad_value=0.3)
            grid_np = grid.permute(1, 2, 0).numpy()

            ax.imshow(grid_np)
            ax.axis('off')
            ax.set_ylabel(f'val = {dim_values[v_idx]:.1f}', fontsize=11, fontweight='bold')

        fig.suptitle(f'Dimension {dim} — Adjusting a single noise value', fontsize=14, y=1.02)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'noise_dim{dim}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'维度 {dim} 实验图已保存: {save_path}')
        plt.show()


def run_experiment(
    pth_path='generator.pth',
    z_dim=100,
    n_images=8,
    base_seed=0,
    chosen_dims=None,
    dim_values=None,
):
    """
    主实验流程。

    参数:
        chosen_dims: 要探索的 5 个维度索引列表，例如 [0, 25, 50, 75, 99]
        dim_values: 每个维度要尝试的 3 个值，例如 [-2.0, 0.0, 2.0]
    """
    # ---- 默认参数 ----
    if chosen_dims is None:
        chosen_dims = [0, 25, 50, 75, 99]
    if dim_values is None:
        dim_values = [-2.0, 0.0, 2.0]

    assert len(chosen_dims) == 5, f'请选择恰好 5 个维度，当前选了 {len(chosen_dims)} 个'
    assert len(dim_values) == 3, f'请提供恰好 3 个取值，当前提供了 {len(dim_values)} 个'

    # ---- 加载模型 ----
    print('加载 Generator...')
    G, device = load_generator(pth_path, z_dim)
    print(f'设备: {device}')

    # ---- 生成 base noise ----
    print(f'使用 base_seed={base_seed} 生成 {n_images} 个 base noise 向量...')
    base_noise = make_noise_batch(n_images, z_dim, base_seed=base_seed, device='cpu')

    # ---- 实验 ----
    # all_results[d][v] = generated images tensor (8, 1, 28, 28)
    all_results = []

    for dim_idx, dim in enumerate(chosen_dims):
        print(f'\n维度 {dim}:')
        dim_results = []
        variants = vary_one_dim(base_noise, dim, dim_values)

        for val, modified_noise in variants:
            imgs = generate_images(G, modified_noise, device)
            dim_results.append(imgs)
            print(f'  noise[{dim}] = {val:6.1f} → 已生成 8 张图')

        all_results.append(dim_results)

    # ---- 绘图 ----
    plot_experiment(all_results, chosen_dims, dim_values)

    # ============================================================
    # 定量分析：计算每个维度变化时图像的差异
    # ============================================================
    print('\n' + '=' * 70)
    print('定量分析：每个维度调整前后，生成图像的像素变化')
    print('=' * 70)

    for d_idx, dim in enumerate(chosen_dims):
        print(f'\n--- 维度 {dim} ---')
        base_imgs = all_results[d_idx][1]  # val = 0.0 作为基准（中间值）

        for v_idx, val in enumerate(dim_values):
            if v_idx == 1:  # 跳过基准自身
                continue
            target_imgs = all_results[d_idx][v_idx]
            # 计算平均像素差异
            diff = (target_imgs - base_imgs).abs().mean().item()
            print(f'  noise[{dim}] 从 {dim_values[1]:.1f} → {val:5.1f}: '
                  f'平均像素变化 = {diff:.4f}')

    return all_results, chosen_dims, dim_values


# ============================================================
# 你也可以自由选择其他维度或其他取值，下面提供几种方案
# ============================================================

def experiment_wide_range():
    """方案 A：在较大范围内调整（适合观察剧烈变化）"""
    chosen_dims = [0, 20, 40, 60, 80]
    dim_values = [-3.0, 0.0, 3.0]
    return run_experiment(chosen_dims=chosen_dims, dim_values=dim_values)


def experiment_fine_grained():
    """方案 B：在较小范围内细调（适合观察微妙变化）"""
    chosen_dims = [5, 30, 55, 70, 95]
    dim_values = [-0.5, 0.0, 0.5]
    return run_experiment(chosen_dims=chosen_dims, dim_values=dim_values)


def experiment_asymmetric():
    """方案 C：不对称取值，观察偏置效应"""
    chosen_dims = [10, 35, 60, 85, 99]
    dim_values = [-1.0, 0.5, 2.0]
    return run_experiment(chosen_dims=chosen_dims, dim_values=dim_values)


# ============================================================
# 辅助：生成不同 seed 的 base noise 看看效果
# ============================================================

def compare_base_seeds():
    """用不同的 base_seed 重复实验，看维度效应是否一致。"""
    chosen_dims = [0, 25, 50, 75, 99]
    dim_values = [-2.0, 0.0, 2.0]

    for seed in [0, 42, 123]:
        print(f'\n{"#"*60}')
        print(f'# base_seed = {seed}')
        print(f'{"#"*60}')
        base_noise = make_noise_batch(8, 100, base_seed=seed, device='cpu')

        # 直接用 run_experiment 但改 base_noise 的逻辑……简化处理，直接复用
        G, device = load_generator()
        all_results = []
        for dim in chosen_dims:
            dim_results = []
            for val in dim_values:
                modified = base_noise.clone()
                modified[:, dim] = val
                imgs = generate_images(G, modified, device)
                dim_results.append(imgs)
            all_results.append(dim_results)

        plot_experiment(
            all_results, chosen_dims, dim_values,
            save_dir=f'seed{seed}',
        )


# ============================================================
# 进阶：固定其他所有值，只让这一个维度变化（极端情况）
# ============================================================

def extreme_test():
    """
    极端测试：将 8 个 base noise 的 99 个维度全部固定为 0，
    只保留 1 个维度变化。看单维度的生成能力。
    """
    print('\n' + '=' * 70)
    print('极端测试：99 维固定为 0，仅 1 维变化')
    print('=' * 70)

    G, device = load_generator()
    n_images = 8
    dims_to_test = [0, 25, 50, 75, 99]
    values_to_test = [-3.0, -1.0, 0.0, 1.0, 3.0]

    fig, axes = plt.subplots(len(dims_to_test), len(values_to_test),
                             figsize=(len(values_to_test) * 3, len(dims_to_test) * 3))

    for d_idx, dim in enumerate(dims_to_test):
        for v_idx, val in enumerate(values_to_test):
            # 全零噪声，只改这一维
            noise = torch.zeros(n_images, 100, device=device)
            noise[:, dim] = val

            with torch.no_grad():
                imgs = G(noise).cpu()

            grid = vutils.make_grid(imgs, nrow=8, normalize=True, pad_value=0.3)
            grid_np = grid.permute(1, 2, 0).numpy()

            ax = axes[d_idx, v_idx]
            ax.imshow(grid_np)
            ax.axis('off')
            if d_idx == 0:
                ax.set_title(f'val={val:.1f}', fontsize=10)
            if v_idx == 0:
                ax.set_ylabel(f'dim={dim}', fontsize=10, fontweight='bold')

    plt.suptitle('Extreme: only one noise dimension active (others = 0)', fontsize=13)
    plt.tight_layout()
    plt.savefig('noise_extreme_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('极端测试图已保存到 noise_extreme_test.png')


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    print('=' * 70)
    print('GAN 噪声维度扰动实验')
    print('=' * 70)
    print('''
实验方案（默认）:
  - 从 100 维噪声中挑选 5 个维度: [0, 25, 50, 75, 99]
  - 每个维度尝试 3 个取值: [-2.0, 0.0, 2.0]
  - 每个设置生成 8 张图 (基于固定 base seed)
  - 总共: 5 × 3 × 8 = 120 张图
    ''')

    # ---- 运行主实验 ----
    run_experiment()

    # ---- 可选：取消注释运行其他实验 ----
    # experiment_wide_range()
    # experiment_fine_grained()
    # compare_base_seeds()
    # extreme_test()

    print('\n' + '=' * 70)
    print('生成结束')
    print('=' * 70)
