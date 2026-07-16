"""
Lab 4.4: Tiny DDPM Diffusion Model from Scratch on 2D Toy Data
================================================================
Objective: Implementasi DDPM minimal dari scratch, training pada 2D point clouds.
           Visualisasi proses noising dan denoising secara intuitif.

PLATFORM NOTES:
- ✓ MacBook Air M3: Berjalan baik di CPU! Dataset 2D sangat kecil.
- ✓ Tidak butuh GPU — ini adalah lab pedagogis, selesai dalam ~5-10 menit di CPU
- ✓ Google Colab: Bahkan lebih cepat

INSTALL:
    pip install torch numpy matplotlib scikit-learn
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_moons, make_swiss_roll
import time
import os

# ============================================================
# 1. CONFIG
# ============================================================
T = 1000            # Jumlah diffusion steps
BETA_START = 1e-4   # Noise schedule start
BETA_END = 0.02     # Noise schedule end
DEVICE = "cpu"      # CPU cukup untuk 2D toy data
HIDDEN_DIM = 128
EPOCHS = 5000
BATCH_SIZE = 256
LR = 1e-3
SEED = 42
OUTPUT_DIR = "./diffusion_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("Lab 4.4: Tiny DDPM on 2D Toy Data")
print(f"Device: {DEVICE} | T: {T} | Epochs: {EPOCHS}")
print("=" * 60)

# ============================================================
# 2. TOY DATASETS
# ============================================================
def generate_data(dataset_type="moons", n_samples=10000):
    """Generate berbagai tipe 2D toy data."""
    if dataset_type == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=0.05)
    elif dataset_type == "gaussians":
        # 4 Gaussian dalam grid
        centers = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        X = np.concatenate([
            np.random.randn(n_samples // 4, 2) * 0.3 + c
            for c in centers
        ])
    elif dataset_type == "spiral":
        n = n_samples // 2
        theta = np.linspace(0, 4*np.pi, n)
        r = np.linspace(0.1, 1, n)
        X1 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        X2 = np.stack([-r * np.cos(theta), -r * np.sin(theta)], axis=1)
        X = np.concatenate([X1, X2])
    elif dataset_type == "circle":
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        r = 1.0 + np.random.randn(n_samples) * 0.1
        X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")

    # Normalize ke [-2, 2]
    X = (X - X.mean(0)) / X.std(0)
    return torch.tensor(X, dtype=torch.float32)

# Gunakan "moons" default
data = generate_data("moons", n_samples=10000).to(DEVICE)
print(f"✓ Data shape: {data.shape} | Range: [{data.min():.2f}, {data.max():.2f}]")

# ============================================================
# 3. DIFFUSION SCHEDULE (Beta Schedule)
# ============================================================
betas = torch.linspace(BETA_START, BETA_END, T, device=DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

print(f"✓ Noise schedule: β₁={BETA_START:.4f}, β_T={BETA_END:.4f}")
print(f"  √ᾱ_T = {sqrt_alphas_cumprod[-1]:.4f} (mendekati 0 = pure noise)")

# ============================================================
# 4. FORWARD DIFFUSION PROCESS
# ============================================================
def q_sample(x0, t, noise=None):
    """
    Forward diffusion: sample x_t dari x_0
    q(x_t | x_0) = N(√ᾱ_t · x_0, (1 - ᾱ_t) · I)
    Closed form: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    return sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise, noise

# Visualisasi forward process
def visualize_forward_process(data, save_path=None):
    """Visualisasi bagaimana data di-noised secara bertahap."""
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    timesteps = [0, T//5, 2*T//5, 3*T//5, 4*T//5, T-1]
    n_vis = min(2000, len(data))

    for i, t_val in enumerate(timesteps):
        if t_val == 0:
            x_vis = data[:n_vis].cpu().numpy()
        else:
            t_tensor = torch.full((n_vis,), t_val, device=DEVICE, dtype=torch.long)
            x_vis, _ = q_sample(data[:n_vis], t_tensor)
            x_vis = x_vis.cpu().numpy()

        axes[i].scatter(x_vis[:, 0], x_vis[:, 1], s=2, alpha=0.5, c='steelblue')
        axes[i].set_title(f't = {t_val}', fontsize=10)
        axes[i].set_xlim(-4, 4)
        axes[i].set_ylim(-4, 4)
        axes[i].set_aspect('equal')
        axes[i].axis('off')

    axes[0].set_title('t=0\n(Clean Data)', fontsize=10)
    axes[-1].set_title(f't={T-1}\n(Pure Noise)', fontsize=10)
    plt.suptitle('Forward Diffusion Process: Data → Noise', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    plt.show()

print("\n[Visualizing forward process...]")
visualize_forward_process(data, save_path=f"{OUTPUT_DIR}/forward_process.png")

# ============================================================
# 5. DENOISING MODEL (MLP dengan Time Embedding)
# ============================================================
class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time embedding seperti di Transformer."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class NoisePredictionMLP(nn.Module):
    """
    MLP yang memprediksi noise ε di timestep t.
    Input: x_t (2D point), t (timestep)
    Output: predicted noise ε (2D)
    """
    def __init__(self, input_dim=2, hidden_dim=128, time_dim=32):
        super().__init__()
        self.time_embed = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        # Main network
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),   # Output: predicted noise (2D)
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        x_cat = torch.cat([x, t_emb], dim=-1)
        return self.net(x_cat)

model = NoisePredictionMLP(hidden_dim=HIDDEN_DIM).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n✓ Model dibuat: {n_params:,} trainable parameters")

# ============================================================
# 6. TRAINING LOOP
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

loss_history = []
print(f"\n[Training {EPOCHS} epochs...]")
t_start = time.time()

for epoch in range(EPOCHS):
    # Sample random batch
    idx = torch.randint(0, len(data), (BATCH_SIZE,))
    x0 = data[idx]

    # Sample random timestep
    t_batch = torch.randint(0, T, (BATCH_SIZE,), device=DEVICE)

    # Forward diffusion: tambahkan noise
    xt, noise = q_sample(x0, t_batch)

    # Prediksi noise
    noise_pred = model(xt, t_batch)

    # MSE loss antara noise asli dan prediksi
    loss = nn.functional.mse_loss(noise_pred, noise)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())

    if epoch % 500 == 0:
        elapsed = time.time() - t_start
        eta = elapsed / (epoch + 1) * (EPOCHS - epoch - 1)
        print(f"  Epoch {epoch:5d}/{EPOCHS} | Loss: {loss.item():.6f} | ETA: {eta:.0f}s")

print(f"\n✓ Training selesai dalam {time.time() - t_start:.1f}s")

# Plot loss curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Smoothed loss
window = 100
smoothed = np.convolve(loss_history, np.ones(window)/window, mode='valid')
plt.plot(smoothed, color='orange', label=f'Smoothed (window={window})')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (Log)")
plt.title("Smoothed Loss")
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_loss.png", dpi=150)
print(f"✓ Saved: {OUTPUT_DIR}/training_loss.png")
plt.show()

# ============================================================
# 7. REVERSE DIFFUSION (SAMPLING)
# ============================================================
@torch.no_grad()
def p_sample(model, x, t_val):
    """
    Single step reverse diffusion:
    p_θ(x_{t-1} | x_t)
    """
    t_tensor = torch.full((x.shape[0],), t_val, device=DEVICE, dtype=torch.long)
    noise_pred = model(x, t_tensor)

    alpha_t = alphas[t_val]
    alpha_bar_t = alphas_cumprod[t_val]
    beta_t = betas[t_val]

    # Mean of posterior p_θ(x_{t-1}|x_t)
    coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
    mean = (1 / torch.sqrt(alpha_t)) * (x - coef * noise_pred)

    if t_val > 0:
        sigma_t = torch.sqrt(beta_t)
        noise = torch.randn_like(x)
        return mean + sigma_t * noise
    else:
        return mean

@torch.no_grad()
def sample(model, n_samples=2000, verbose=True):
    """Generate samples via reverse diffusion dari noise."""
    model.eval()
    # Mulai dari pure noise
    x = torch.randn(n_samples, 2, device=DEVICE)

    # Simpan intermediate states untuk visualisasi
    snapshots = {T-1: x.clone().cpu().numpy()}
    snap_steps = [3*T//4, T//2, T//4, T//8, 0]

    for t_val in reversed(range(T)):
        x = p_sample(model, x, t_val)
        if t_val in snap_steps:
            snapshots[t_val] = x.clone().cpu().numpy()
        if verbose and t_val % 200 == 0:
            print(f"  Denoising step {T - t_val}/{T}...")

    return x.cpu().numpy(), snapshots

print("\n[Generating samples via reverse diffusion...]")
samples, snapshots = sample(model, n_samples=2000)

# ============================================================
# 8. FINAL VISUALIZATION
# ============================================================
def final_visualization(data, samples, snapshots):
    fig = plt.figure(figsize=(20, 8))
    gs_outer = gridspec.GridSpec(2, 1, hspace=0.4)

    # Baris 1: Forward process
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_outer[0], wspace=0.1)
    timesteps_vis = [0, T//5, 2*T//5, 3*T//5, 4*T//5, T-1]
    colors_fwd = plt.cm.Blues(np.linspace(0.3, 1.0, 6))

    for i, t_val in enumerate(timesteps_vis):
        ax = fig.add_subplot(gs_top[i])
        if t_val == 0:
            x_vis = data[:2000].cpu().numpy()
        else:
            t_tensor = torch.full((2000,), t_val, device=DEVICE, dtype=torch.long)
            x_vis, _ = q_sample(data[:2000], t_tensor)
            x_vis = x_vis.cpu().numpy()
        ax.scatter(x_vis[:, 0], x_vis[:, 1], s=1.5, alpha=0.4, color=colors_fwd[i])
        ax.set_title(f't={t_val}', fontsize=8)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect('equal'); ax.axis('off')
    fig.text(0.5, 0.92, '⬇ Forward Diffusion: Data → Noise', ha='center', fontsize=12, fontweight='bold')

    # Baris 2: Reverse process
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_outer[1], wspace=0.1)
    snap_keys = sorted(snapshots.keys(), reverse=True)
    colors_rev = plt.cm.Greens(np.linspace(0.3, 1.0, 6))

    for i, (k, c) in enumerate(zip(snap_keys, colors_rev)):
        ax = fig.add_subplot(gs_bot[i])
        pts = snapshots[k]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, alpha=0.4, color=c)
        ax.set_title(f't={k}', fontsize=8)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect('equal'); ax.axis('off')
    fig.text(0.5, 0.48, '⬇ Reverse Diffusion: Noise → Generated Data', ha='center', fontsize=12, fontweight='bold')

    plt.savefig(f"{OUTPUT_DIR}/diffusion_complete.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/diffusion_complete.png")
    plt.show()

def compare_real_vs_generated(data, samples):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(data[:2000, 0].cpu(), data[:2000, 1].cpu(), s=3, alpha=0.5, c='steelblue')
    axes[0].set_title("Original Data (Two Moons)", fontsize=12, fontweight='bold')
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    noised_mid, _ = q_sample(data[:2000], torch.full((2000,), T//2, device=DEVICE))
    axes[1].scatter(noised_mid[:, 0].cpu(), noised_mid[:, 1].cpu(), s=3, alpha=0.5, c='orange')
    axes[1].set_title(f"Noised Data (t={T//2})", fontsize=12, fontweight='bold')
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

    axes[2].scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5, c='green')
    axes[2].set_title("Generated Samples (Denoised from Noise)", fontsize=12, fontweight='bold')
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)

    plt.suptitle('DDPM 2D Toy Data — Real vs Noised vs Generated', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/real_vs_generated.png", dpi=150)
    print(f"✓ Saved: {OUTPUT_DIR}/real_vs_generated.png")
    plt.show()

print("\n[Visualizing results...]")
final_visualization(data, samples, snapshots)
compare_real_vs_generated(data, samples)

# ============================================================
# ACTIVITY ANSWERS
# ============================================================
print("\n" + "="*60)
print("ACTIVITY ANSWERS — Lab 4.4")
print("="*60)

answers = """
Q1: Visualisasi data pada berbagai timestep
    t=0    → Data original, Two Moons terlihat jelas (dua cluster melengkung)
    t=T/4  → Struktur masih terlihat namun mulai kabur, noise mulai dominan
    t=T/2  → Batas antara dua moon sangat blur, tapi masih ada hint struktur
    t=3T/4 → Hampir tidak bisa membedakan mana data asli, mendekati Gaussian
    t=T    → Pure Gaussian noise, struktur Two Moons hilang sepenuhnya

Q2: Apa yang terjadi jika T dikurangi dari 1000 ke 100?
    - Noise schedule menjadi lebih "tiba-tiba" (setiap step lebih besar)
    - Model belajar denoising step yang lebih kasar
    - Kualitas sampel menurun — transisi kurang smooth
    - Training lebih cepat tapi sampling quality lebih rendah
    - T=100 masih bisa menghasilkan sampel yang reasonable untuk 2D data
    - Untuk image diffusion, T rendah biasanya butuh teknik khusus (DDIM, DPM++)

Q3: MLP 2-layer vs 4-layer — apakah lebih dalam lebih baik?
    - 2-layer: Training cepat, mungkin underfitting untuk distribusi kompleks
    - 4-layer: Loss biasanya lebih rendah, sampel lebih akurat mengikuti distribusi
    - Diminishing returns setelah 4-6 layer untuk 2D toy data
    - Untuk 2D moons: 3-4 layer sudah cukup baik
    - Untuk image diffusion: butuh UNet yang jauh lebih dalam

Q4: DDIM (Denoising Diffusion Implicit Models)
    DDIM (Song et al., 2020) mempercepat sampling dengan reformulasi:
    - DDPM: Markovian chain, butuh sampling setiap dari T ke 0 (1000 steps)
    - DDIM: Non-Markovian, bisa skip steps — hanya butuh 10-50 steps
    Caranya: Rederive reverse process menggunakan implicit probabilistic model
    yang konsisten dengan forward process tapi tidak terikat ke Markovian chain.
    Hasilnya: sampling 10-50× lebih cepat dengan kualitas hampir sama.
    Persamaan DDIM update:
        x_{t-1} = √ᾱ_{t-1} · x_pred_0 + √(1-ᾱ_{t-1}) · epsilon_theta
    di mana x_pred_0 = (x_t - √(1-ᾱ_t) · epsilon_theta) / √ᾱ_t
"""
print(answers)

print(f"\n✓ Lab 4.4 selesai!")
print(f"Output tersimpan di: {OUTPUT_DIR}/")
print(f"  - forward_process.png")
print(f"  - training_loss.png")
print(f"  - diffusion_complete.png")
print(f"  - real_vs_generated.png")