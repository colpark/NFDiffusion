"""
Coordinate-Conditioned Diffusion v2 (Fixed) - CIFAR-10 Training Script

Improvements over v1:
1. PixelShuffle upsampling (eliminates checkerboard artifacts)
2. 16 coordinate frequencies (was 10 - better high-freq details)
3. Scale 8.0 (was 10.0 - less aliasing)
4. Bicubic sparse interpolation (was bilinear - smoother)

Train once on 32×32, infer at 32×32/64×64/96×96 WITHOUT artifacts!
"""

import os, math, random, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision as tv
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

# =========================
# Configuration
# =========================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
IMG_SIZE    = 32
CHANNELS    = 3
MODEL_DIM   = 64
TIMESTEPS   = 1000

# Coordinate encoding (v2 fixed)
COORD_NUM_FREQ = 16  # Increased from 10
COORD_SCALE    = 8.0  # Reduced from 10.0

# Data
SPARSITY    = 0.40
COND_FRAC   = 0.50
MASK_SEED   = 12345

BATCH_TRAIN = 128  # Larger batch for 50K samples
BATCH_TEST  = 64
NUM_WORKERS = 4

# Training (longer for CIFAR-10)
STEPS       = 100_000  # 100K steps for 50K training samples
PRINT_EVERY = 100
SAVE_EVERY  = 5_000
VIS_EVERY   = 2000

# Evaluation
N_VIS           = 8
DDIM_STEPS_VIS  = 50
DDIM_STEPS_EVAL = 50
MAX_TEST_BATCHES= 5

# Directories
DATA_ROOT   = "./data_cifar10"
CKPT_DIR    = "./kevinref_inc_1/cifar10_v2_ckpts"
OUT_DIR     = "./kevinref_inc_1/cifar10_v2_outputs"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Coordinate encoding: {COORD_NUM_FREQ} frequencies, scale {COORD_SCALE}")

# =========================
# Coordinate Encoding
# =========================
class FourierCoordinateEncoding(nn.Module):
    def __init__(self, num_frequencies=16, scale=8.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.scale = scale
        self.encoding_dim = 4 * num_frequencies

    def forward(self, coords):
        B, H, W, _ = coords.shape
        x = coords[..., 0:1]
        y = coords[..., 1:2]

        freq_bands = 2.0 ** torch.arange(self.num_frequencies, device=coords.device, dtype=torch.float32)
        freq_bands = freq_bands * math.pi * self.scale

        x_freq = x * freq_bands.view(1, 1, 1, -1)
        x_features = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)

        y_freq = y * freq_bands.view(1, 1, 1, -1)
        y_features = torch.cat([torch.sin(y_freq), torch.cos(y_freq)], dim=-1)

        features = torch.cat([x_features, y_features], dim=-1)
        return features


def make_coordinate_grid(batch_size, height, width, device):
    y_coords = torch.linspace(0, 1, height, device=device)
    x_coords = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1)
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return coords

# =========================
# Utilities
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor):
        if t.dtype != torch.float32:
            t = t.float()
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(0, half, device=device).float() / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


def hw_to_seq(t):
    return t.flatten(2).transpose(1, 2)


def seq_to_hw(t, h, w):
    return t.transpose(1, 2).reshape(t.size(0), -1, h, w)


@torch.no_grad()
def soft_project(x, obs, mask, iters=1):
    for _ in range(iters):
        x = x * (1.0 - mask) + obs * mask
    return x


def save_grid01(tensors01, path, nrow=6, pad=2):
    rows = []
    for t in tensors01:
        grid = make_grid(t, nrow=nrow, padding=pad)
        rows.append(grid)
    big = torch.cat(rows, dim=1)
    save_image(big, path)

# =========================
# UNet Components (v2 Fixed)
# =========================
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=None, groups=32):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(self.norm1, self.activation1, self.conv1)

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if time_emb_dim is not None else None

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=dim_out)
        self.activation2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        self.block2 = nn.Sequential(self.norm2, self.activation2, self.dropout, self.conv2)

        self.residual_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if time_emb is not None and self.mlp is not None:
            h = h + self.mlp(time_emb)[..., None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, groups=32):
        super().__init__()
        self.dim = dim
        self.scale = dim ** (-0.5)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.to_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = [hw_to_seq(t) for t in qkv]
        sim = torch.einsum('bic,bjc->bij', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bij,bjc->bic', attn, v)
        out = seq_to_hw(out, h, w)
        return self.to_out(out) + x


class ResnetAttentionBlock(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=None, groups=32):
        super().__init__()
        self.resnet = ResnetBlock(dim, dim_out, time_emb_dim, dropout, groups)
        self.attention = Attention(dim_out if dim_out is not None else dim, groups)

    def forward(self, x, time_emb=None):
        x = self.resnet(x, time_emb)
        return self.attention(x)


class downSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.downsample = nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.downsample(x)


class upSample(nn.Module):
    """v2 Fixed: PixelShuffle upsampling to eliminate checkerboard artifacts."""
    def __init__(self, dim_in):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.upsample(x)


class CoordinateConditionedUnet(nn.Module):
    def __init__(self, dim, image_size, dim_multiply=(1, 2, 4, 8), channel=3, num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.0, device='cuda', groups=32,
                 coord_num_frequencies=16, coord_scale=8.0):
        super().__init__()
        assert dim % groups == 0

        self.dim = dim
        self.channel = channel
        self.time_emb_dim = 4 * self.dim
        self.num_resolutions = len(dim_multiply)
        self.device = device
        self.resolution = [int(image_size / (2 ** i)) for i in range(self.num_resolutions)]
        self.hidden_dims = [self.dim, *map(lambda x: x * self.dim, dim_multiply)]
        self.num_res_blocks = num_res_blocks

        positional_encoding = PositionalEncoding(self.dim)
        self.time_mlp = nn.Sequential(
            positional_encoding, nn.Linear(self.dim, self.time_emb_dim),
            nn.SiLU(), nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        self.coord_encoder = FourierCoordinateEncoding(
            num_frequencies=coord_num_frequencies,
            scale=coord_scale
        )
        coord_dim = self.coord_encoder.encoding_dim

        self.down_path = nn.ModuleList([])
        self.up_path = nn.ModuleList([])
        concat_dim = []

        self.init_conv = nn.Conv2d(channel * 3 + coord_dim, self.dim, kernel_size=3, padding=1)
        concat_dim.append(self.dim)

        for level in range(self.num_resolutions):
            d_in, d_out = self.hidden_dims[level], self.hidden_dims[level + 1]
            for block in range(num_res_blocks):
                d_in_ = d_in if block == 0 else d_out
                if self.resolution[level] in attn_resolutions:
                    self.down_path.append(ResnetAttentionBlock(d_in_, d_out, self.time_emb_dim, dropout, groups))
                else:
                    self.down_path.append(ResnetBlock(d_in_, d_out, self.time_emb_dim, dropout, groups))
                concat_dim.append(d_out)
            if level != self.num_resolutions - 1:
                self.down_path.append(downSample(d_out))
                concat_dim.append(d_out)

        mid_dim = self.hidden_dims[-1]
        self.middle_resnet_attention = ResnetAttentionBlock(mid_dim, mid_dim, self.time_emb_dim, dropout, groups)
        self.middle_resnet = ResnetBlock(mid_dim, mid_dim, self.time_emb_dim, dropout, groups)

        for level in reversed(range(self.num_resolutions)):
            d_out = self.hidden_dims[level + 1]
            for block in range(num_res_blocks + 1):
                d_in = self.hidden_dims[level + 2] if block == 0 and level != self.num_resolutions - 1 else d_out
                d_in = d_in + concat_dim.pop()
                if self.resolution[level] in attn_resolutions:
                    self.up_path.append(ResnetAttentionBlock(d_in, d_out, self.time_emb_dim, dropout, groups))
                else:
                    self.up_path.append(ResnetBlock(d_in, d_out, self.time_emb_dim, dropout, groups))
            if level != 0:
                self.up_path.append(upSample(d_out))  # Now uses PixelShuffle!

        assert not concat_dim

        final_ch = self.hidden_dims[1]
        self.final_norm = nn.GroupNorm(groups, final_ch)
        self.final_activation = nn.SiLU()
        self.final_conv = nn.Conv2d(final_ch, channel, kernel_size=3, padding=1)

    def forward(self, x, time, sparse_input=None, mask=None, coords=None, x_coarse=None):
        B, C, H, W = x.shape

        if coords is None:
            coords = make_coordinate_grid(B, H, W, x.device)

        coord_features = self.coord_encoder(coords)
        coord_features = coord_features.permute(0, 3, 1, 2)

        t = self.time_mlp(time)
        x_with_coords = torch.cat([x, coord_features], dim=1)

        concat = []
        x = self.init_conv(x_with_coords)
        concat.append(x)

        for layer in self.down_path:
            if isinstance(layer, (upSample, downSample)):
                x = layer(x)
            else:
                x = layer(x, t)
            concat.append(x)

        x = self.middle_resnet_attention(x, t)
        x = self.middle_resnet(x, t)

        for layer in self.up_path:
            if not isinstance(layer, upSample):
                x = torch.cat((x, concat.pop()), dim=1)
            if isinstance(layer, (upSample, downSample)):
                x = layer(x)
            else:
                x = layer(x, t)

        x = self.final_activation(self.final_norm(x))
        return self.final_conv(x)

# =========================
# Diffusion
# =========================
class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, time_step=1000, loss_type='l2'):
        super().__init__()
        self.unet = model
        self.channel = self.unet.channel
        self.device = next(self.unet.parameters()).device
        self.image_size = image_size
        self.time_step = time_step
        self.loss_type = loss_type

        beta = self.linear_beta_schedule()
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], pad=(1, 0), value=1.)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)

        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - alpha_bar))

        self.register_buffer('beta_tilde', beta * ((1. - alpha_bar_prev) / (1. - alpha_bar)))
        self.register_buffer('mean_tilde_x0_coeff', beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar))
        self.register_buffer('mean_tilde_xt_coeff', torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

        self.register_buffer('sqrt_recip_alpha_bar', torch.sqrt(1. / alpha_bar))
        self.register_buffer('sqrt_recip_alpha_bar_min_1', torch.sqrt(1. / alpha_bar - 1))
        self.register_buffer('sqrt_recip_alpha', torch.sqrt(1. / alpha))
        self.register_buffer('beta_over_sqrt_one_minus_alpha_bar', beta / torch.sqrt(1. - alpha_bar))

    def q_sample(self, x0, t, noise):
        return self.sqrt_alpha_bar[t][:, None, None, None] * x0 + \
               self.sqrt_one_minus_alpha_bar[t][:, None, None, None] * noise

    def forward(self, img, sparse_input=None, mask=None, coords=None, perceiver_input=None, loss_mask=None):
        b, c, h, w = img.shape

        def _match_channels(t, target_C):
            if t.size(1) == target_C:
                return t
            if t.size(1) == 1 and target_C > 1:
                return t.repeat(1, target_C, 1, 1)
            if target_C == 1 and t.size(1) > 1:
                return t.mean(dim=1, keepdim=True)
            raise RuntimeError(f"Channel mismatch: have {t.size(1)}, need {target_C}")

        t = torch.randint(0, self.time_step, (b,), device=img.device).long()
        noise = torch.randn_like(img)
        noised_image = self.q_sample(img, t, noise)

        if sparse_input is not None and mask is not None:
            model_input = torch.cat([noised_image, sparse_input, mask], dim=1)
        else:
            model_input = noised_image

        if coords is None:
            coords = make_coordinate_grid(b, h, w, img.device)

        predicted_noise = self.unet(model_input, t, coords=coords, x_coarse=perceiver_input)

        if predicted_noise.size(1) != noise.size(1):
            ref_C = max(predicted_noise.size(1), noise.size(1))
            noise           = _match_channels(noise,           ref_C)
            predicted_noise = _match_channels(predicted_noise, ref_C)

        if mask is not None and mask.size(1) != predicted_noise.size(1):
            mask = _match_channels(mask, predicted_noise.size(1))
        if loss_mask is not None and loss_mask.size(1) != predicted_noise.size(1):
            loss_mask = _match_channels(loss_mask, predicted_noise.size(1))

        if self.loss_type == 'l1':
            raw_loss = F.l1_loss(noise, predicted_noise, reduction='none')
        elif self.loss_type == 'l2':
            raw_loss = F.mse_loss(noise, predicted_noise, reduction='none')
        elif self.loss_type == "huber":
            raw_loss = F.smooth_l1_loss(noise, predicted_noise, reduction='none')
        else:
            raise NotImplementedError()

        if loss_mask is not None:
            lambda_cond = 0.05
            combined_mask = (loss_mask + lambda_cond * mask).clamp(max=1.0)
            loss = (raw_loss * combined_mask).sum() / combined_mask.sum().clamp_min(1e-8)
        else:
            loss = raw_loss.mean()
        return loss

    def linear_beta_schedule(self):
        scale = 1000 / self.time_step
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.time_step, dtype=torch.float32)


class DDIM_Sampler(nn.Module):
    def __init__(self, ddpm_diffusion_model, ddim_steps=50, eta=0.0, clip=True):
        super().__init__()
        self.model = ddpm_diffusion_model
        self.ddim_steps = int(ddim_steps)
        self.eta = float(eta)
        self.clip = clip

        with torch.no_grad():
            ab = self.model.alpha_bar
            self.register_buffer('tau',
                torch.linspace(0, self.model.time_step-1, steps=self.ddim_steps, dtype=torch.long))
            alpha_tau = ab[self.tau]
            alpha_prev = F.pad(alpha_tau[:-1], (1,0), value=1.0)
            self.register_buffer('alpha_tau', alpha_tau)
            self.register_buffer('alpha_prev', alpha_prev)

            sig = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_tau) * (1 - alpha_tau / alpha_prev))
            coeff = torch.sqrt(1 - alpha_prev - sig**2)
            self.register_buffer('sigma', sig)
            self.register_buffer('coeff', coeff)
            self.register_buffer('sqrt_alpha_prev', torch.sqrt(alpha_prev))

    @torch.inference_mode()
    def sample(self, batch_size, sparse_input, mask, target_size=None, min1to1=True):
        device = self.model.device
        C = self.model.channel

        if target_size is None:
            H, W = sparse_input.shape[2:]
        elif isinstance(target_size, int):
            H = W = target_size
        else:
            H, W = target_size

        # v2 Fixed: Use bicubic interpolation
        if (H, W) != sparse_input.shape[2:]:
            sparse_target = F.interpolate(sparse_input, size=(H, W), mode='bicubic', align_corners=False)
            mask_target = F.interpolate(mask, size=(H, W), mode='nearest')
        else:
            sparse_target = sparse_input
            mask_target = mask

        coords = make_coordinate_grid(batch_size, H, W, device)
        xt = torch.randn([batch_size, C, H, W], device=device)
        xt = xt * (1.0 - mask_target) + sparse_target * mask_target
        xt = soft_project(xt, sparse_target, mask_target, iters=1)

        for i in reversed(range(self.ddim_steps)):
            t = self.tau[i]
            bt = torch.full((batch_size,), t, device=device, dtype=torch.long)

            sp = sparse_target if sparse_target.size(1)==C else sparse_target.repeat(1,C,1,1)
            mk = mask_target if mask_target.size(1)==C else mask_target.repeat(1,C,1,1)

            model_in = torch.cat([xt, sp, mk], dim=1)
            pred_eps = self.model.unet(model_in, bt, coords=coords)

            x0 = self.model.sqrt_recip_alpha_bar[t] * xt - self.model.sqrt_recip_alpha_bar_min_1[t] * pred_eps
            if self.clip:
                x0.clamp_(-1., 1.)
                pred_eps = (self.model.sqrt_recip_alpha_bar[t] * xt - x0) / self.model.sqrt_recip_alpha_bar_min_1[t]

            mean = self.sqrt_alpha_prev[i] * x0 + self.coeff[i] * pred_eps
            noise = torch.randn_like(xt) if i > 0 else 0.
            xt = mean + self.sigma[i] * noise
            xt = xt * (1.0 - mk) + sp * mk

        xt.clamp_(min=-1.0, max=1.0)
        return (xt + 1.0)/2.0 if min1to1 else xt

# =========================
# Dataset
# =========================
class CIFAR10_32(Dataset):
    def __init__(self, root, split='train', augment=True):
        tfms = []
        if augment and split == 'train':
            tfms += [T.RandomHorizontalFlip()]
        tfms += [
            T.ToTensor(),
            T.Lambda(lambda t: t*2.0 - 1.0),
        ]
        self.tf = T.Compose(tfms)
        train_flag = (split == 'train')
        self.ds = tv.datasets.CIFAR10(root=root, train=train_flag, download=True, transform=self.tf)
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        return x


class FixedMaskWrapper(Dataset):
    def __init__(self, base_ds: Dataset, sparsity: float, cond_frac: float, seed: int):
        self.base = base_ds
        self.sparsity = sparsity
        self.cond_frac = cond_frac
        self.seed = seed
        n = len(self.base)
        x0 = self.base[0]
        _, H, W = x0.shape
        self.m_cond_1 = torch.zeros(n, 1, H, W, dtype=torch.float32)
        self.m_supv_1 = torch.zeros(n, 1, H, W, dtype=torch.float32)
        print(f"Generating fixed masks for {n} samples...")
        for i in range(n):
            if i % 5000 == 0:
                print(f"  Progress: {i}/{n}")
            g = torch.Generator().manual_seed(seed + i)
            m_full = (torch.rand(1, H, W, generator=g) < sparsity).float()
            coords = m_full.nonzero(as_tuple=False)
            if coords.numel() > 0:
                idx_hw = coords[:,1]*W + coords[:,2]
                perm = torch.randperm(idx_hw.numel(), generator=g)
                k = int(round(cond_frac * idx_hw.numel()))
                take = idx_hw[perm[:k]]
                rr = (take // W).long()
                cc = (take %  W).long()
                m_cond = torch.zeros(1, H, W)
                m_cond[0, rr, cc] = 1.0
                m_supv = m_full - m_cond
            else:
                m_cond = torch.zeros(1, H, W)
                m_supv = torch.zeros(1, H, W)
            self.m_cond_1[i] = m_cond
            self.m_supv_1[i] = m_supv
        print("Mask generation complete.")
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x = self.base[idx]
        return x, self.m_cond_1[idx], self.m_supv_1[idx]

# =========================
# Main Training
# =========================
def main():
    # Build model
    def attn_resolutions_for(image_size, levels=(1,), dim_mult=(1,2,2,2)):
        res_list = [int(image_size / (2 ** i)) for i in range(len(dim_mult))]
        return tuple(res_list[i] for i in levels)

    attn_res = attn_resolutions_for(IMG_SIZE, (1,))

    net = CoordinateConditionedUnet(
        dim=MODEL_DIM,
        image_size=IMG_SIZE,
        dim_multiply=(1,2,2,2),
        channel=CHANNELS,
        num_res_blocks=2,
        attn_resolutions=attn_res,
        dropout=0.0,
        device=DEVICE,
        groups=32,
        coord_num_frequencies=COORD_NUM_FREQ,
        coord_scale=COORD_SCALE
    ).to(DEVICE)

    ddpm = GaussianDiffusion(
        model=net,
        image_size=IMG_SIZE,
        time_step=TIMESTEPS,
        loss_type='l2'
    ).to(DEVICE)

    opt = torch.optim.AdamW(ddpm.parameters(), lr=2e-4, weight_decay=1e-4)

    print(f"Model built with {sum(p.numel() for p in net.parameters())/1e6:.2f}M parameters")

    # Prepare data
    base_train = CIFAR10_32(DATA_ROOT, split='train', augment=True)
    train_ds   = FixedMaskWrapper(base_train, sparsity=SPARSITY, cond_frac=COND_FRAC, seed=MASK_SEED)

    base_test = CIFAR10_32(DATA_ROOT, split='test', augment=False)
    test_ds   = FixedMaskWrapper(base_test, sparsity=SPARSITY, cond_frac=COND_FRAC, seed=MASK_SEED+777)

    train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    vis_batch = next(iter(test_dl))
    vis_x, vis_m_cond_1, _ = [v[:N_VIS] for v in vis_batch]

    print(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Training loop
    ddpm.train()
    ddim_vis = DDIM_Sampler(ddpm, ddim_steps=DDIM_STEPS_VIS, eta=0.0, clip=True)

    it = iter(train_dl)
    for step in range(1, STEPS + 1):
        try:
            x, m_cond_1, m_supv_1 = next(it)
        except StopIteration:
            it = iter(train_dl)
            x, m_cond_1, m_supv_1 = next(it)

        x = x.to(DEVICE)
        m_cond_1 = m_cond_1.to(DEVICE)
        m_supv_1 = m_supv_1.to(DEVICE)

        B, C, H, W = x.shape
        m_cond = m_cond_1.repeat(1, C, 1, 1)
        m_supv = m_supv_1.repeat(1, C, 1, 1)
        x_sparse = x * m_cond

        coords = make_coordinate_grid(B, H, W, DEVICE)

        loss = ddpm(img=x, sparse_input=x_sparse, mask=m_cond, coords=coords,
                    perceiver_input=None, loss_mask=m_supv)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
        opt.step()

        if step % PRINT_EVERY == 0:
            print(f"[{step}/{STEPS}] loss={loss.item():.4f}")

        if step % VIS_EVERY == 0 or step == 1:
            ddpm.eval()
            with torch.inference_mode():
                vx = vis_x.to(DEVICE)
                vm = vis_m_cond_1.to(DEVICE)
                vm3 = vm.repeat(1, CHANNELS, 1, 1)
                v_sparse = vx * vm3

                recon01 = ddim_vis.sample(batch_size=vx.size(0),
                                          sparse_input=v_sparse,
                                          mask=vm3,
                                          target_size=32,
                                          min1to1=True)

                gt01 = (vx + 1.0)/2.0
                sp01 = (v_sparse + 1.0)/2.0
                out_path = os.path.join(OUT_DIR, f"train_vis_step{step}_32x32.png")
                save_grid01([gt01.cpu(), sp01.cpu(), recon01.cpu()], out_path, nrow=min(N_VIS, 8))
                print(f"Saved vis → {out_path}")
            ddpm.train()

        if step % SAVE_EVERY == 0 or step == STEPS:
            torch.save({'net': net.state_dict()}, os.path.join(CKPT_DIR, f"net_step{step}.pt"))
            print(f"Checkpoint saved at step {step}")

    print("Training complete!")

    # Evaluation
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50)

    ckpt_path = os.path.join(CKPT_DIR, f"net_step{STEPS}.pt")
    state = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(state['net'], strict=True)
    ddpm.eval()

    ddim_eval = DDIM_Sampler(ddpm, ddim_steps=DDIM_STEPS_EVAL, eta=0.0, clip=True)

    test_batch = next(iter(test_dl))
    x_test, m_cond_1_test, _ = test_batch
    x_test = x_test[:N_VIS].to(DEVICE)
    m_cond_1_test = m_cond_1_test[:N_VIS].to(DEVICE)

    m_cond_test = m_cond_1_test.repeat(1, CHANNELS, 1, 1)
    x_sparse_test = x_test * m_cond_test

    print("Testing zero-shot super-resolution at multiple resolutions...")

    with torch.inference_mode():
        print("Reconstructing at 32×32...")
        recon_32 = ddim_eval.sample(
            batch_size=x_test.size(0),
            sparse_input=x_sparse_test,
            mask=m_cond_test,
            target_size=32,
            min1to1=True
        )

        print("Zero-shot super-resolution to 64×64...")
        recon_64 = ddim_eval.sample(
            batch_size=x_test.size(0),
            sparse_input=x_sparse_test,
            mask=m_cond_test,
            target_size=64,
            min1to1=True
        )

        print("Zero-shot super-resolution to 96×96...")
        recon_96 = ddim_eval.sample(
            batch_size=x_test.size(0),
            sparse_input=x_sparse_test,
            mask=m_cond_test,
            target_size=96,
            min1to1=True
        )

    gt01 = (x_test + 1.0) / 2.0
    sp01 = (x_sparse_test + 1.0) / 2.0

    gt64 = F.interpolate(gt01, size=(64, 64), mode='bicubic', align_corners=False)
    gt96 = F.interpolate(gt01, size=(96, 96), mode='bicubic', align_corners=False)
    sp64 = F.interpolate(sp01, size=(64, 64), mode='bicubic', align_corners=False)
    sp96 = F.interpolate(sp01, size=(96, 96), mode='bicubic', align_corners=False)

    save_grid01([gt01.cpu(), sp01.cpu(), recon_32.cpu()],
                os.path.join(OUT_DIR, "eval_32x32.png"), nrow=N_VIS)
    save_grid01([gt64.cpu(), sp64.cpu(), recon_64.cpu()],
                os.path.join(OUT_DIR, "eval_64x64.png"), nrow=N_VIS)
    save_grid01([gt96.cpu(), sp96.cpu(), recon_96.cpu()],
                os.path.join(OUT_DIR, "eval_96x96.png"), nrow=N_VIS)

    print("\nResults saved:")
    print(f"  - {OUT_DIR}/eval_32x32.png (native resolution)")
    print(f"  - {OUT_DIR}/eval_64x64.png (zero-shot 2× super-resolution)")
    print(f"  - {OUT_DIR}/eval_96x96.png (zero-shot 3× super-resolution)")

    # Compute PSNR
    def psnr_batch(pred, target):
        mse = F.mse_loss(pred, target, reduction='mean')
        if mse < 1e-10:
            return 99.0
        return 10.0 * math.log10(1.0 / mse.item())

    psnr_32 = psnr_batch(recon_32, gt01)
    psnr_64 = psnr_batch(recon_64, gt64)
    psnr_96 = psnr_batch(recon_96, gt96)

    print(f"\nPSNR Results (on {N_VIS} test samples):")
    print(f"  32×32: {psnr_32:.2f} dB")
    print(f"  64×64: {psnr_64:.2f} dB (zero-shot)")
    print(f"  96×96: {psnr_96:.2f} dB (zero-shot)")

    print("\n" + "="*50)
    print("Done! Check outputs in:", OUT_DIR)
    print("="*50)


if __name__ == "__main__":
    main()
