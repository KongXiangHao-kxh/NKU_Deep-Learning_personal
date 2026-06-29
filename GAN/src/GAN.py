import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# ============================================================
# 1.  Model Definitions
# ============================================================

class Discriminator(nn.Module):
    """Binary classifier: real (1) vs fake (0)."""
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # (B, 1, 28, 28) -> (B, 784)
        x = x.view(x.size(0), 784)
        h = self.nonlin1(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out


class Generator(nn.Module):
    """Maps noise z ~ N(0,1) to fake image (1,28,28)."""
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = torch.tanh(self.fc2(h))          # range [-1, 1]
        out = out.view(out.size(0), 1, 28, 28)  # image shape
        return out


# ============================================================
# 2.  Helper: show image grids
# ============================================================

def show_imgs(x, new_fig=True, title=None):
    """Display a grid of generated images."""
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.permute(1, 2, 0).numpy()
    if new_fig:
        plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()


# ============================================================
# 3.  Training
# ============================================================

def train_gan(
    n_epochs=20,
    batch_size=64,
    z_dim=100,
    lr_d=0.03,
    lr_g=0.03,
    device=None,
    data_root='./FashionMNIST',
    print_every=200,
):
    """Train the original GAN on FashionMNIST and return loss history + models."""

    # ---- Device ----
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ---- Data ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # scale to [-1, 1]
    ])
    dataset = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    # ---- Models ----
    D = Discriminator().to(device)
    G = Generator(z_dim=z_dim).to(device)

    print('=' * 50)
    print('Discriminator architecture:')
    print('=' * 50)
    print(D)
    print()

    print('=' * 50)
    print('Generator architecture:')
    print('=' * 50)
    print(G)
    print()

    # ---- Optimisers & loss ----
    optimizerD = torch.optim.SGD(D.parameters(), lr=lr_d)
    optimizerG = torch.optim.SGD(G.parameters(), lr=lr_g)
    criterion = nn.BCELoss()

    lab_real = torch.full((batch_size, 1), 1.0, device=device)
    lab_fake = torch.full((batch_size, 1), 0.0, device=device)

    # Fixed noise for visualisation throughout training
    fixed_noise = torch.randn(batch_size, z_dim, device=device)

    # ---- Logging ----
    history = {
        'lossD': [],      # discriminator loss (real + fake)
        'lossG': [],      # generator loss (-log D(G(z)))
        'D_x':    [],      # mean D(x_real)
        'D_G_z':  [],      # mean D(G(z)) for generator
    }
    gen_samples = []  # list of generated image grids (one per epoch)

    n_batches = len(dataloader)
    print(f'Training on {n_batches} batches/epoch × {n_epochs} epochs\n')

    # ---- Training loop ----
    for epoch in range(n_epochs):
        for i, (x_real, _) in enumerate(dataloader):
            x_real = x_real.to(device)
            current_bs = x_real.size(0)

            # -------- Step 1: Discriminator --------
            optimizerD.zero_grad()

            # Real images: D(x) -> 1
            D_x = D(x_real)
            lossD_real = criterion(D_x, lab_real[:current_bs])

            # Fake images: D(G(z)) -> 0
            z = torch.randn(current_bs, z_dim, device=device)
            x_gen = G(z).detach()           # detach so G does not get gradients here
            D_G_z_fake = D(x_gen)
            lossD_fake = criterion(D_G_z_fake, lab_fake[:current_bs])

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # -------- Step 2: Generator (non-saturating loss) --------
            optimizerG.zero_grad()

            z = torch.randn(current_bs, z_dim, device=device)
            x_gen = G(z)
            D_G_z = D(x_gen)
            lossG = criterion(D_G_z, lab_real[:current_bs])  # -log D(G(z))

            lossG.backward()
            optimizerG.step()

            # -------- Logging --------
            history['lossD'].append(lossD.item())
            history['lossG'].append(lossG.item())
            history['D_x'].append(D_x.mean().item())
            history['D_G_z'].append(D_G_z.mean().item())

            if i % print_every == 0:
                print(
                    f'epoch {epoch:2d} | batch {i:4d}/{n_batches} | '
                    f'lossD={lossD.item():.4f} lossG={lossG.item():.4f} | '
                    f'D(x)={D_x.mean().item():.4f} D(G(z))={D_G_z.mean().item():.4f}'
                )

        # End-of-epoch: generate samples with fixed noise
        with torch.no_grad():
            samples = G(fixed_noise).detach().cpu()
        gen_samples.append(samples)

    print('\nTraining complete.')
    return D, G, history, gen_samples


# ============================================================
# 4.  Plotting
# ============================================================

def plot_training_curves(history):
    """Plot the loss curves for D and G, plus the D(x) / D(G(z)) scores."""
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Losses
    ax = axes[0]
    ax.plot(history['lossD'], label='Discriminator loss', alpha=0.7, linewidth=0.8)
    ax.plot(history['lossG'], label='Generator loss', alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Training step')
    ax.set_ylabel('BCE Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: D(x) and D(G(z)) scores
    ax = axes[1]
    ax.plot(history['D_x'], label='D(x) — real', alpha=0.7, linewidth=0.8)
    ax.plot(history['D_G_z'], label="D(G(z)) — fake", alpha=0.7, linewidth=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Probability')
    ax.set_title('Discriminator Output Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gan_training_curves.png', dpi=150)
    plt.show()
    print('Saved training curves to gan_training_curves.png')


def plot_generated_samples(gen_samples):
    """Display generated samples from each epoch side by side."""
    n_epochs = len(gen_samples)
    _, axes = plt.subplots(1, n_epochs, figsize=(5 * n_epochs, 5))
    if n_epochs == 1:
        axes = [axes]
    for ax, samples, ep in zip(axes, gen_samples, range(n_epochs)):
        grid = vutils.make_grid(samples, nrow=8, normalize=True, pad_value=0.3)
        grid = grid.permute(1, 2, 0).numpy()
        ax.imshow(grid)
        ax.set_title(f'Epoch {ep}')
        ax.axis('off')
    plt.suptitle('Generator samples (fixed noise) over training', fontsize=14)
    plt.tight_layout()
    plt.savefig('gan_generated_samples.png', dpi=150)
    plt.show()
    print('Saved generated samples to gan_generated_samples.png')


def print_model_parameter_count(D, G):
    """Print parameter count for both models."""
    d_params = sum(p.numel() for p in D.parameters())
    g_params = sum(p.numel() for p in G.parameters())
    print('=' * 50)
    print('Parameter count')
    print('=' * 50)
    print(f'Discriminator: {d_params:,} parameters')
    print(f'Generator:     {g_params:,} parameters')
    print(f'Total:         {d_params + g_params:,} parameters')
    print()

    # Detailed layer-by-layer
    print('Discriminator details:')
    for name, p in D.named_parameters():
        print(f'  {name}: {list(p.shape)} → {p.numel():,} params')
    print()
    print('Generator details:')
    for name, p in G.named_parameters():
        print(f'  {name}: {list(p.shape)} → {p.numel():,} params')
    print()


# ============================================================
# 5.  Main
# ============================================================

if __name__ == '__main__':
    # --- Train ---
    D, G, history, gen_samples = train_gan(
        n_epochs=20,
        batch_size=64,
        z_dim=100,
        lr_d=0.03,
        lr_g=0.03,
        print_every=200,
    )

    # --- Parameter count ---
    print_model_parameter_count(D, G)

    # --- Plot ---
    plot_training_curves(history)
    plot_generated_samples(gen_samples)

    # --- Save models ---
    torch.save(D.state_dict(), 'discriminator.pth')
    torch.save(G.state_dict(), 'generator.pth')
    print('Saved model weights to discriminator.pth / generator.pth')
