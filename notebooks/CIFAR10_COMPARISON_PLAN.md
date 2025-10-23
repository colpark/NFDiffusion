# CIFAR-10 Sparse Reconstruction: 3-Way Comparison Plan

## Overview

Compare three single-step training approaches for sparse-to-sparse CIFAR-10 reconstruction:
- **Input**: 20% randomly sampled pixels (instance-specific)
- **Output**: Different 20% randomly sampled pixels (non-overlapping)
- **Backbone**: Perceiver IO with Fourier features
- **Evaluation**: Unified metrics across all approaches

---

## Shared Components

### 1. Data: `core/sparse/cifar10_sparse.py`
```python
SparseCIFAR10Dataset(
    input_ratio=0.2,   # 20% input pixels (~204 pixels)
    output_ratio=0.2,  # 20% output pixels (~204 pixels, non-overlapping)
    seed=42            # Reproducible instance-specific sampling
)
```

**Returns per sample**:
- `input_coords`: (204, 2) normalized [0,1] coordinates
- `input_values`: (204, 3) RGB values
- `output_coords`: (204, 2) coordinates for prediction
- `output_values`: (204, 3) target RGB values
- `full_image`: (3, 32, 32) for evaluation

### 2. Architecture: `core/neural_fields/perceiver.py`
```python
PerceiverIO(
    input_channels=3,
    output_channels=3,
    num_latents=512,
    latent_dim=512,
    num_fourier_feats=256,
    num_blocks=6,
    num_heads=8
)
```

**Fourier Features**: γ(x,y) = [sin(2πBp), cos(2πBp)] where B ~ N(0, scale²I)

### 3. Metrics: `core/sparse/metrics.py`
- **Pixel-level**: MSE, MAE on output pixels
- **Image-level**: PSNR, SSIM on reconstructed images
- **MetricsTracker**: Unified tracking across all approaches

---

## Approach 1: Score-Based Neural Field

### **Notebook**: `03_cifar10_score_based.ipynb`

### Theory
Learn score function s_θ(x, coords, t | input) = ∇_x log p_t(x | input)

- **Forward SDE**: dx = -βt/2 · x dt + √βt dw
- **Reverse SDE**: dx = [-βt/2 · x - βt · s_θ(x,t)] dt + √βt dw
- **Training loss**: Denoising score matching

### Architecture
```python
class ScoreNetwork(nn.Module):
    def __init__(self):
        self.perceiver = PerceiverIO(...)
        self.time_embed = SinusoidalEmbedding(dim=512)

    def forward(self, noisy_values, output_coords, t, input_coords, input_values):
        # Embed time
        t_emb = self.time_embed(t)  # (B, 512)

        # Condition: concatenate input pixels with noisy output pixels + time
        cond_coords = torch.cat([input_coords, output_coords], dim=1)
        cond_values = torch.cat([input_values, noisy_values], dim=1)

        # Add time embedding to values
        cond_values = cond_values + t_emb.unsqueeze(1)

        # Predict score at output coordinates
        score = self.perceiver(cond_coords, cond_values, output_coords)
        return score
```

### Training
```python
# Sample timestep t ~ Uniform(0, T)
# Add noise: x_t = α_t · x_0 + σ_t · ε, ε ~ N(0, I)
# Predict score: s_θ(x_t, coords, t | input)
# Loss: ||s_θ + ε/σ_t||²
```

### Sampling (Langevin Dynamics)
```python
x_T ~ N(0, I)
for t in reversed(range(T)):
    z ~ N(0, I)
    score = score_network(x_t, coords, t, input)
    x_{t-1} = x_t + step_size * score + √(2 * step_size) * z
```

---

## Approach 2: Neural Field as Denoiser (Gaussian Random Field)

### **Notebook**: `04_cifar10_nf_denoiser.ipynb`

### Theory (Your Gaussian RF Idea!)
Start with Gaussian noise at all output pixel locations, progressively denoise conditioned on input pixels.

- **Process**: Noisy field → Neural field denoise → Less noisy field → ... → Clean field
- **No score matching**: Direct denoising objective
- **Single network**: Predicts clean values from noisy values at each step

### Architecture
```python
class NFDenoiser(nn.Module):
    def __init__(self):
        self.perceiver = PerceiverIO(...)
        self.time_embed = SinusoidalEmbedding(dim=512)
        self.noise_encoder = nn.Linear(3, 512)  # Encode noise level

    def forward(self, noisy_values, output_coords, t, input_coords, input_values):
        # Encode noise level
        noise_emb = self.noise_encoder(torch.full((B, 1, 3), fill_value=t))

        # Condition on input pixels (fixed throughout diffusion)
        # Denoise output pixels (changing at each step)
        all_coords = torch.cat([input_coords, output_coords], dim=1)
        all_values = torch.cat([input_values, noisy_values + noise_emb], dim=1)

        # Predict clean values at output coords
        clean_values = self.perceiver(all_coords, all_values, output_coords)
        return clean_values
```

### Training
```python
# Sample timestep t ~ Uniform(0, T)
# Add noise: x_t = √(α_t) · x_0 + √(1 - α_t) · ε
# Predict clean: x̂_0 = denoiser(x_t, coords, t | input)
# Loss: ||x̂_0 - x_0||²
```

### Sampling (DDPM/DDIM)
```python
x_T ~ N(0, I)  # Gaussian random field at output locations
for t in reversed(range(T)):
    x_0_pred = denoiser(x_t, coords, t, input)
    # Compute x_{t-1} using DDPM or DDIM update rule
    x_{t-1} = ...
```

**Key difference**: Input pixels remain fixed (clean), only output pixels are noisy and denoised!

---

## Approach 3: Flow Matching

### **Notebook**: `05_cifar10_flow_matching.ipynb`

### Theory
Learn vector field v_θ that transports p_0 (noise) → p_1 (data)

- **Forward flow**: dx/dt = v_θ(x_t, coords, t | input)
- **Conditional flow matching**: φ_t(x_1) = (1-t)·x_0 + t·x_1
- **Training loss**: ||v_θ(φ_t(x_1), t) - (x_1 - x_0)||²

### Architecture
```python
class VelocityField(nn.Module):
    def __init__(self):
        self.perceiver = PerceiverIO(...)
        self.time_embed = SinusoidalEmbedding(dim=512)

    def forward(self, x_t, output_coords, t, input_coords, input_values):
        # Time embedding
        t_emb = self.time_embed(t)

        # Condition on input + current state
        all_coords = torch.cat([input_coords, output_coords], dim=1)
        all_values = torch.cat([input_values, x_t], dim=1)

        # Add time to values
        all_values = all_values + t_emb.unsqueeze(1)

        # Predict velocity
        velocity = self.perceiver(all_coords, all_values, output_coords)
        return velocity
```

### Training
```python
# Sample t ~ Uniform(0, 1)
# Sample x_0 ~ N(0, I), x_1 = data
# Compute φ_t = (1-t)·x_0 + t·x_1
# Predict velocity: v_θ(φ_t, coords, t | input)
# Target velocity: x_1 - x_0
# Loss: ||v_θ - (x_1 - x_0)||²
```

### Sampling (ODE Solver)
```python
x_0 ~ N(0, I)
# Solve ODE: dx/dt = v_θ(x_t, coords, t | input) from t=0 to t=1
# Use Euler, Heun, or Runge-Kutta solver
for t in linspace(0, 1, num_steps):
    velocity = velocity_field(x_t, coords, t, input)
    x_{t+dt} = x_t + dt * velocity
```

---

## Unified Evaluation Framework

### Training Metrics (per epoch)
```python
tracker = MetricsTracker()
for batch in train_loader:
    pred = model.sample(input_coords, input_values, output_coords)
    tracker.update(pred, target_values)

results = tracker.compute()
# MSE, MAE, PSNR, SSIM
```

### Test Metrics (final evaluation)
```python
# 1. Output pixel reconstruction
- MSE on 204 output pixels
- MAE on 204 output pixels

# 2. Full image quality (if reconstructed)
- PSNR (dB)
- SSIM (structural similarity)

# 3. Visual quality
- Input + Predicted vs Ground Truth
- Error heatmaps
```

### Comparison Table
| Metric | Score-Based | NF Denoiser | Flow Matching |
|--------|-------------|-------------|---------------|
| MSE | ? | ? | ? |
| MAE | ? | ? | ? |
| PSNR | ? | ? | ? |
| SSIM | ? | ? | ? |
| Training Time | ? | ? | ? |
| Sampling Time | ? | ? | ? |
| Parameters | ~15M | ~15M | ~15M |

### Arbitrary Resolution Test
All three approaches can query at arbitrary coordinates:
```python
# Super-resolution: 32x32 → 64x64
hr_coords = create_grid(64, 64)
hr_pred = model.sample(input_coords, input_values, hr_coords)
```

---

## Implementation Details

### Hyperparameters (consistent across all)
```python
# Model
num_latents = 512
latent_dim = 512
num_fourier_feats = 256
num_blocks = 6
num_heads = 8

# Training
batch_size = 64
epochs = 100
lr = 1e-4
diffusion_steps = 100

# Dataset
input_ratio = 0.2
output_ratio = 0.2
seed = 42  # For reproducibility
```

### Training Schedule
```python
# All use cosine schedule for noise/time
beta_schedule = cosine_schedule(timesteps=100)

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

---

## Expected Outcomes

### Approach Strengths

**Score-Based**:
- ✅ Principled probabilistic framework
- ✅ Well-studied theory
- ⚠️ Langevin sampling slower

**NF Denoiser**:
- ✅ Most intuitive (your Gaussian RF idea!)
- ✅ Direct prediction objective
- ✅ Cleanest separation: input=fixed, output=noisy

**Flow Matching**:
- ✅ Fastest sampling (straight paths)
- ✅ Simple training objective
- ✅ Modern approach (SD3 uses this)

### Research Questions
1. Which approach best handles sparse conditioning?
2. How do sampling speeds compare in practice?
3. Quality vs. speed trade-offs?
4. Generalization to different sparsity ratios?

---

## Next Steps After Comparison

1. **Best performer**: Optimize further (architecture, hyperparams)
2. **Ablations**: Fourier features, Perceiver depth, latent count
3. **Extensions**: Variable sparsity, different sampling patterns
4. **Applications**: Super-resolution, inpainting, temporal interpolation

---

Ready to implement? Shall I proceed with creating all 3 notebooks?
