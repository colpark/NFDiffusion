# Sparse reconstruction of STL-10 @ 32×32 (no super-resolution)
# Adds: (1) periodic training visualization with DDIM
#       (2) fast capped evaluation with DDIM to avoid stalling

import os, math, random, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision as tv
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# =========================
# Utilities (self-contained)
# =========================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional/time embedding -> (B, dim) given integer timesteps."""
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
        return emb  # (B, dim)

def hw_to_seq(t):  # (B, C, H, W) -> (B, HW, C)
    return t.flatten(2).transpose(1, 2)

def seq_to_hw(t, h, w):  # (B, HW, C) -> (B, C, H, W)
    return t.transpose(1, 2).reshape(t.size(0), -1, h, w)

@torch.no_grad()
def soft_project(x, obs, mask, kernel_size=3, iters=1):
    # Minimal projection: enforce observed pixels exactly
    for _ in range(iters):
        x = x * (1.0 - mask) + obs * mask
    return x

def to_img01(t):
    return ((t.clamp(-1,1) + 1.0)/2.0).detach().cpu()

def save_grid01(tensors01, path, nrow=6, pad=2):
    """
    tensors01: list of [B,3,H,W] in [0,1] (same B).
    Saves a vertical stack of grids (GT / Sparse / Recon).
    """
    rows = []
    for t in tensors01:
        grid = make_grid(t, nrow=nrow, padding=pad)
        rows.append(grid)
    big = torch.cat(rows, dim=1)  # stack vertically
    save_image(big, path)

# =========================
# UNet (self-contained)
# =========================
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=None, groups=32):
        super().__init__()
        self.dim, self.dim_out = dim, dim_out
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
        q, k, v = [hw_to_seq(t) for t in qkv]  # (B, HW, C)
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
        self.downsameple = nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.downsameple(x)

class upSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1))
    def forward(self, x):
        return self.upsample(x)

class Unet(nn.Module):
    def __init__(self, dim, image_size, dim_multiply=(1, 2, 4, 8), channel=3, num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.0, device='cuda', groups=32):
        super().__init__()
        assert dim % groups == 0, 'parameter [groups] must be divisible by parameter [dim]'

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

        self.down_path = nn.ModuleList([])
        self.up_path = nn.ModuleList([])
        concat_dim = []

        # Input is concatenated [xt, sparse, mask] -> channel*3
        self.init_conv = nn.Conv2d(channel * 3, self.dim, kernel_size=3, padding=1)
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
                self.up_path.append(upSample(d_out))

        assert not concat_dim, 'Error in concatenation between downward path and upward path.'

        final_ch = self.hidden_dims[1]
        self.final_norm = nn.GroupNorm(groups, final_ch)
        self.final_activation = nn.SiLU()
        self.final_conv = nn.Conv2d(final_ch, channel, kernel_size=3, padding=1)

    def forward(self, x, time, sparse_input=None, mask=None, x_coarse=None):
        t = self.time_mlp(time)

        concat = []
        x = self.init_conv(x)
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
# Diffusion + DDIM
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

    def forward(self, img, sparse_input=None, mask=None, perceiver_input=None, loss_mask=None):
        b, c, h, w = img.shape

        def _match_channels(t, target_C):
            if t.size(1) == target_C:
                return t
            if t.size(1) == 1 and target_C > 1:
                return t.repeat(1, target_C, 1, 1)
            if target_C == 1 and t.size(1) > 1:
                return t.mean(dim=1, keepdim=True)
            raise RuntimeError(f"Channel mismatch: have {t.size(1)}, need {target_C}")

        assert h == self.image_size and w == self.image_size, f'height and width of image must be {self.image_size}'
        t = torch.randint(0, self.time_step, (b,), device=img.device).long()
        noise = torch.randn_like(img)
        noised_image = self.q_sample(img, t, noise)

        if sparse_input is not None and mask is not None:
            model_input = torch.cat([noised_image, sparse_input, mask], dim=1)
        else:
            model_input = noised_image

        predicted_noise = self.unet(model_input, t, x_coarse=perceiver_input)

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

    @torch.inference_mode()
    def p_sample(self, xt, t, clip=True, sparse_input=None, perceiver_input=None, mask=None):
        batched_time = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)

        C = xt.size(1)
        if sparse_input is not None and sparse_input.size(1) != C:
            sparse_input = sparse_input.repeat(1, C, 1, 1) if sparse_input.size(1) == 1 else sparse_input[:, :C]
        if mask is not None and mask.size(1) != C:
            mask = mask.repeat(1, C, 1, 1) if mask.size(1) == 1 else mask[:, :C]

        if sparse_input is not None and mask is not None:
            model_input = torch.cat([xt, sparse_input, mask], dim=1)
        else:
            model_input = xt

        pred_noise = self.unet(model_input, batched_time, x_coarse=perceiver_input)

        if clip:
            x0 = self.sqrt_recip_alpha_bar[t] * xt - self.sqrt_recip_alpha_bar_min_1[t] * pred_noise
            x0.clamp_(-1., 1.)
            mean = self.mean_tilde_x0_coeff[t] * x0 + self.mean_tilde_xt_coeff[t] * xt
        else:
            mean = self.sqrt_recip_alpha[t] * (xt - self.beta_over_sqrt_one_minus_alpha_bar[t] * pred_noise)
        variance = self.beta_tilde[t]
        noise = torch.randn_like(xt) if t > 0 else 0.
        return mean + torch.sqrt(variance) * noise

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timestep=False, clip=True, min1to1=False,
               sparse_input=None, perceiver_input=None, mask=None):

        xT = torch.randn([batch_size, self.channel, self.image_size, self.image_size], device=self.device)

        # Start consistent: enforce observed pixels
        if sparse_input is not None and mask is not None:
            xT = xT * (1.0 - mask) + sparse_input * mask

        assert sparse_input is not None and mask is not None, \
            "Must provide sparse_input and mask for conditioned sampling."

        denoised_intermediates = [xT]
        xt = xT

        xt = soft_project(xt, sparse_input, mask, kernel_size=3)
        for t in reversed(range(0, self.time_step)):
            x_t_minus_1 = self.p_sample(xt, t, clip, sparse_input, perceiver_input, mask)
            x_t_minus_1 = x_t_minus_1 * (1.0 - mask) + sparse_input * mask
            denoised_intermediates.append(x_t_minus_1)
            xt = x_t_minus_1

        images = xt if not return_all_timestep else torch.stack(denoised_intermediates, dim=1)
        images.clamp_(min=-1.0, max=1.0)
        if not min1to1:
            images = (images + 1.0) / 2.0
        return images

    def linear_beta_schedule(self):
        scale = 1000 / self.time_step
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.time_step, dtype=torch.float32)

class DDIM_Sampler(nn.Module):
    """
    Lightweight DDIM using the ddpm model's alpha_bar schedule.
    """
    def __init__(self, ddpm_diffusion_model, ddim_steps=50, eta=0.0, clip=True):
        super().__init__()
        self.model = ddpm_diffusion_model
        self.ddim_steps = int(ddim_steps)
        self.eta = float(eta)
        self.clip = clip

        with torch.no_grad():
            ab = self.model.alpha_bar
            # choose timestep indices uniformly in [0, T-1]
            self.register_buffer('tau',
                torch.linspace(0, self.model.time_step-1, steps=self.ddim_steps, dtype=torch.long))
            alpha_tau = ab[self.tau]
            alpha_prev = F.pad(alpha_tau[:-1], (1,0), value=1.0)
            self.register_buffer('alpha_tau', alpha_tau)
            self.register_buffer('alpha_prev', alpha_prev)

            # sigma_t from DDIM paper
            sig = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_tau) * (1 - alpha_tau / alpha_prev))
            coeff = torch.sqrt(1 - alpha_prev - sig**2)
            self.register_buffer('sigma', sig)
            self.register_buffer('coeff', coeff)
            self.register_buffer('sqrt_alpha_prev', torch.sqrt(alpha_prev))

    @torch.inference_mode()
    def sample(self, batch_size, sparse_input, mask, min1to1=True):
        device = self.model.device
        C, H, W = self.model.channel, self.model.image_size, self.model.image_size
        xt = torch.randn([batch_size, C, H, W], device=device)

        # Start consistent with observations
        xt = xt * (1.0 - mask) + sparse_input * mask
        xt = soft_project(xt, sparse_input, mask, iters=1)

        for i in reversed(range(self.ddim_steps)):
            t = self.tau[i]
            bt = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Ensure channel match of conds
            sp = sparse_input if sparse_input.size(1)==C else sparse_input.repeat(1,C,1,1)
            mk = mask         if mask.size(1)==C         else mask.repeat(1,C,1,1)

            model_in = torch.cat([xt, sp, mk], dim=1)
            pred_eps = self.model.unet(model_in, bt)

            # x0 pred
            x0 = self.model.sqrt_recip_alpha_bar[t] * xt - self.model.sqrt_recip_alpha_bar_min_1[t] * pred_eps
            if self.clip:
                x0.clamp_(-1., 1.)
                pred_eps = (self.model.sqrt_recip_alpha_bar[t] * xt - x0) / self.model.sqrt_recip_alpha_bar_min_1[t]

            mean = self.sqrt_alpha_prev[i] * x0 + self.coeff[i] * pred_eps
            noise = torch.randn_like(xt) if i > 0 else 0.
            xt = mean + self.sigma[i] * noise

            # Re-project to observation manifold
            xt = xt * (1.0 - mk) + sp * mk

        xt.clamp_(min=-1.0, max=1.0)
        return (xt + 1.0)/2.0 if min1to1 else xt

# =========================
# Data + fixed per-item masks
# =========================
class STL10_32(Dataset):
    def __init__(self, root, split='train', augment=True):
        tfms = []
        if augment and split == 'train':
            tfms += [T.RandomHorizontalFlip()]
        tfms += [
            T.Resize((32,32), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Lambda(lambda t: t*2.0 - 1.0),   # [-1,1]
        ]
        self.tf = T.Compose(tfms)
        self.ds = tv.datasets.STL10(root=root, split=split, download=True, transform=self.tf)
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        return x  # [3,32,32] in [-1,1]

class FixedMaskWrapper(Dataset):
    """Persistent per-item masks with fixed sparsity and 50/50 cond/supervision split."""
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
        for i in range(n):
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
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x = self.base[idx]
        return x, self.m_cond_1[idx], self.m_supv_1[idx]

# =========================
# Metrics
# =========================
def psnr_from_minus1_1(x, y):
    x01 = (x + 1.0) / 2.0
    y01 = (y + 1.0) / 2.0
    mse = F.mse_loss(x01, y01, reduction='mean').item()
    if mse <= 1e-10: return 99.0
    return 10.0 * math.log10(1.0 / mse)

def rmse(x, y):
    return torch.sqrt(F.mse_loss(x, y, reduction='mean')).item()

# =========================
# Hyperparameters
# =========================
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG32       = 32
CHANNELS    = 3
MODEL_DIM   = 64
TIMESTEPS   = 1000

SPARSITY    = 0.40
COND_FRAC   = 0.50
MASK_SEED   = 12345

BATCH_TRAIN = 64
BATCH_TEST  = 32
NUM_WORKERS = 4

STEPS       = 100_000  # bump for real training
PRINT_EVERY = 100
SAVE_EVERY  = 2_000

# NEW: visualization & eval speeds
VIS_EVERY       = 1000   # visualize every N steps
N_VIS           = 6      # number of examples in grid
DDIM_STEPS_VIS  = 50     # fast DDIM sample steps for vis
DDIM_STEPS_EVAL = 50     # fast DDIM sample steps for eval
MAX_TEST_BATCHES= 5      # cap eval to avoid long runs

DATA_ROOT   = "./data_stl10"
CKPT_DIR    = "./stl10_recon32_ckpts"
OUT_DIR     = "./stl10_recon32_outputs"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Attention scale helper
def attn_resolutions_for(image_size, levels=(1,), dim_mult=(1,2,2,2)):
    res_list = [int(image_size / (2 ** i)) for i in range(len(dim_mult))]
    return tuple(res_list[i] for i in levels)

ATTN_LEVELS = (1,)  # 16× for 32×32 at level idx 1
attn32 = attn_resolutions_for(IMG32, ATTN_LEVELS)

# =========================
# Build model @ 32×32
# =========================
net32 = Unet(
    dim=MODEL_DIM, image_size=IMG32, dim_multiply=(1,2,2,2),
    channel=CHANNELS, num_res_blocks=2, attn_resolutions=attn32,
    dropout=0.0, device=DEVICE, groups=32
).to(DEVICE)

ddpm32 = GaussianDiffusion(model=net32, image_size=IMG32, time_step=TIMESTEPS, loss_type='l2').to(DEVICE)
opt = torch.optim.AdamW(ddpm32.parameters(), lr=2e-4, weight_decay=1e-4)

# =========================
# Data loaders (train/test both at 32×32 with masks)
# =========================
base_train = STL10_32(DATA_ROOT, split='train', augment=True)
train_ds   = FixedMaskWrapper(base_train, sparsity=SPARSITY, cond_frac=COND_FRAC, seed=MASK_SEED)

base_test = STL10_32(DATA_ROOT, split='test', augment=False)
test_ds   = FixedMaskWrapper(base_test, sparsity=SPARSITY, cond_frac=COND_FRAC, seed=MASK_SEED+777)

train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False,
                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

# Prepare a small, fixed vis batch (so results are comparable across steps)
vis_batch = next(iter(test_dl))
vis_x, vis_m_cond_1, _ = [v[:N_VIS] for v in vis_batch]  # first N_VIS examples

# =========================
# Train @ 32×32 (sparse reconstruction) + periodic vis
# =========================
ddpm32.train()
ddim_vis = DDIM_Sampler(ddpm32, ddim_steps=DDIM_STEPS_VIS, eta=0.0, clip=True)

it = iter(train_dl)
for step in range(1, STEPS + 1):
    try:
        x, m_cond_1, m_supv_1 = next(it)
    except StopIteration:
        it = iter(train_dl); x, m_cond_1, m_supv_1 = next(it)

    x = x.to(DEVICE)                       # [B,3,32,32]
    m_cond_1 = m_cond_1.to(DEVICE)         # [B,1,32,32]
    m_supv_1 = m_supv_1.to(DEVICE)         # [B,1,32,32]

    B, C, H, W = x.shape
    m_cond = m_cond_1.repeat(1, C, 1, 1)   # [B,3,H,W]
    m_supv = m_supv_1.repeat(1, C, 1, 1)   # [B,3,H,W]
    x_sparse = x * m_cond

    loss = ddpm32(img=x, sparse_input=x_sparse, mask=m_cond, perceiver_input=None, loss_mask=m_supv)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(ddpm32.parameters(), 1.0)
    opt.step()

    if step % PRINT_EVERY == 0:
        print(f"[{step}/{STEPS}] loss={loss.item():.4f}  sparsity={SPARSITY:.3f}")

    # Periodic visualization (fast DDIM; small batch)
    if step % VIS_EVERY == 0 or step == 1:
        ddpm32.eval()
        with torch.inference_mode():
            vx = vis_x.to(DEVICE)                                       # [-1,1]
            vm = vis_m_cond_1.to(DEVICE)                                # [B,1,H,W]
            vm3 = vm.repeat(1, CHANNELS, 1, 1)
            v_sparse = vx * vm3
            recon01 = ddim_vis.sample(batch_size=vx.size(0),
                                      sparse_input=v_sparse, mask=vm3, min1to1=True)
            gt01 = (vx + 1.0)/2.0
            sp01 = (v_sparse + 1.0)/2.0
            out_path = os.path.join(OUT_DIR, f"train_vis_step{step}.png")
            save_grid01([gt01.cpu(), sp01.cpu(), recon01.cpu()], out_path, nrow=min(N_VIS, 6))
            print(f"Saved vis → {out_path}")
        ddpm32.train()

    if step % SAVE_EVERY == 0 or step == STEPS:
        torch.save({'net': net32.state_dict()}, os.path.join(CKPT_DIR, f"net32_step{step}.pt"))

# =========================
# Evaluate: fast DDIM reconstruction on capped test set (@32×32)
# =========================
ckpt_path = os.path.join(CKPT_DIR, f"net32_step{STEPS}.pt")
state = torch.load(ckpt_path, map_location='cpu')
net32.load_state_dict(state['net'], strict=True)
ddpm32.eval()

ddim_eval = DDIM_Sampler(ddpm32, ddim_steps=DDIM_STEPS_EVAL, eta=0.0, clip=True)

@torch.no_grad()
def reconstruct_batch_ddim(x, m_cond_1):
    # x: [-1,1], m_cond_1: [B,1,H,W]
    B, C, H, W = x.shape
    m_cond = m_cond_1.to(x.device).repeat(1, C, 1, 1)
    x_sparse = x * m_cond
    y01 = ddim_eval.sample(batch_size=B, sparse_input=x_sparse, mask=m_cond, min1to1=True)
    x01 = (x + 1.0) / 2.0
    return y01, x01, x_sparse

def psnr_rmse_batch01(y01, x01):
    ps = []; rs = []
    for i in range(y01.size(0)):
        # PSNR on [0,1]; RMSE on [-1,1] for parity with earlier prints
        ps.append(10.0 * math.log10(1.0 / max(1e-10, F.mse_loss(y01[i], x01[i], reduction='mean').item())))
        rs.append(torch.sqrt(F.mse_loss((y01[i]*2-1), (x01[i]*2-1), reduction='mean')).item())
    return sum(ps)/len(ps), sum(rs)/len(rs)

all_psnr = []; all_rmse = []
batch0_dumped = False

with torch.inference_mode():
    for b, (x, m_cond_1, m_supv_1) in enumerate(test_dl):
        if b >= MAX_TEST_BATCHES:  # cap to avoid long evaluation
            break
        x = x.to(DEVICE)
        y01, gt01, x_sparse = reconstruct_batch_ddim(x, m_cond_1)
        ps, rs = psnr_rmse_batch01(y01, gt01)
        all_psnr.append(ps); all_rmse.append(rs)

        # Save one grid from first processed batch
        if not batch0_dumped:
            vm3 = m_cond_1.to(DEVICE).repeat(1, CHANNELS, 1, 1)
            sp01 = (x_sparse + 1.0)/2.0
            out_path = os.path.join(OUT_DIR, f"eval_vis_batch{b}.png")
            save_grid01([gt01.cpu(), sp01.cpu(), y01.cpu()], out_path, nrow=min(BATCH_TEST, 6))
            print(f"Saved eval vis → {out_path}")
            batch0_dumped = True

mean_psnr = sum(all_psnr)/max(1, len(all_psnr))
mean_rmse = sum(all_rmse)/max(1, len(all_rmse))
print(f"[FAST EVAL DDIM] PSNR(mean over {len(all_psnr)} batches) = {mean_psnr:.2f} dB | "
      f"RMSE(mean) = {mean_rmse:.4f} ([-1,1] space)")
