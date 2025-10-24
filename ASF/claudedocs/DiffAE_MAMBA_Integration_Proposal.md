# DiffAE + MAMBA Integration Proposal

**Date**: 2025-10-24
**Context**: After analyzing DiffAE architecture, proposing integration with MAMBA sparse field diffusion

---

## Quick Summary: What is DiffAE?

**DiffAE = Semantic Encoder + Conditional Diffusion Decoder**

```
Standard Diffusion:  noise → [Diffusion] → image
DiffAE:             image → [Encoder] → semantic latent (512-D)
                                           ↓
                    noise → [Decoder + latent] → image
```

**Key Innovation**: Disentangles semantic content (encoded) from stochastic details (diffused)

**Enables**:
1. High-fidelity reconstruction (PSNR ~30-35 dB)
2. Semantic manipulation (edit smile, age, hair, etc.)
3. Smooth interpolation in latent space
4. Meaningful latent representation

---

## Why This Matters for MAMBA

### Current MAMBA Challenge

**Problem**: Noise and wiggliness in sparse field reconstructions
- Hypothesis: MAMBA's 1D serialization loses 2D spatial structure
- Solution attempted: Multi-directional scanning (4 directions)

**Potential Additional Issue**: No semantic compression
- Current: Diffuse directly in RGB value space (3-D)
- Observation: RGB values are entangled (lighting, texture, color)
- Result: Model learns complex, high-variance mappings

### DiffAE Insight

**Semantic Latent Space** provides:
- **Compression**: 3 RGB values → 512-D semantic features → 3 RGB values
- **Disentanglement**: Separate semantic content from noise
- **Consistency**: Same latent → consistent multi-scale outputs
- **Smoothness**: Latent interpolation smoother than RGB interpolation

---

## Proposed Integration: MAMBA-DiffAE

### Architecture

```python
class MAMBADiffusionAutoencoder(nn.Module):
    """
    Semantic autoencoder for sparse fields using MAMBA

    Workflow:
    1. Encoder: Sparse field → semantic latent
    2. Decoder: Query coords + latent → RGB values
    """

    def __init__(self):
        # ========================================
        # Encoder: Sparse Field → Semantic Latent
        # ========================================
        self.coord_encoder = FourierFeatures(256)
        self.value_proj = nn.Linear(3, 512)

        # Multi-directional MAMBA for 2D spatial awareness
        self.encoder_blocks = nn.ModuleList([
            MultiDirectionalMambaBlock(
                SSMBlockFast, d_model=512, d_state=16
            )
            for _ in range(6)
        ])

        # Pool to fixed-size semantic latent
        self.encoder_pool = nn.AdaptiveAvgPool1d(1)
        self.latent_proj = nn.Linear(512, 512)

        # ========================================
        # Decoder: Latent + Query Coords → RGB
        # ========================================
        self.query_encoder = FourierFeatures(256)
        self.query_proj = nn.Linear(256*2 + 512, 512)  # coords + latent

        # Multi-directional MAMBA decoder
        self.decoder_blocks = nn.ModuleList([
            MultiDirectionalMambaBlock(
                SSMBlockFast, d_model=512, d_state=16
            )
            for _ in range(6)
        ])

        # RGB output head
        self.rgb_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 3)
        )

        # Time embedding for diffusion
        self.time_embed = SinusoidalTimeEmbedding(512)

    def encode(self, input_coords, input_values):
        """
        Encode sparse field to semantic latent

        Args:
            input_coords: (B, N_in, 2)
            input_values: (B, N_in, 3)

        Returns:
            latent: (B, 512)
        """
        # Fourier features + value embedding
        coord_feats = self.coord_encoder(input_coords)  # (B, N_in, 512)
        value_feats = self.value_proj(input_values)     # (B, N_in, 512)
        seq = coord_feats + value_feats                 # (B, N_in, 512)

        # Multi-directional MAMBA encoding
        for block in self.encoder_blocks:
            seq = block(seq, input_coords)  # 4-way spatial processing

        # Pool to fixed-size latent
        latent = self.encoder_pool(seq.transpose(1, 2))  # (B, 512, 1)
        latent = latent.squeeze(-1)                       # (B, 512)
        latent = self.latent_proj(latent)                # (B, 512)

        return latent

    def decode(self, query_coords, latent, t, noisy_values=None):
        """
        Decode latent to RGB at query coordinates (with diffusion)

        Args:
            query_coords: (B, N_out, 2)
            latent: (B, 512) - semantic latent from encoder
            t: (B,) - diffusion timestep
            noisy_values: (B, N_out, 3) - noisy RGB (for diffusion)

        Returns:
            pred_values: (B, N_out, 3) - predicted RGB
        """
        B, N_out = query_coords.shape[:2]

        # Time embedding
        t_emb = self.time_embed(t)  # (B, 512)

        # Query features: coords + latent + noisy values
        query_feats = self.query_encoder(query_coords)  # (B, N_out, 512)
        latent_expanded = latent.unsqueeze(1).expand(B, N_out, 512)

        if noisy_values is not None:
            # During diffusion training
            seq = torch.cat([
                query_feats, latent_expanded, noisy_values
            ], dim=-1)  # (B, N_out, 512+512+3)
            seq = self.query_proj(seq)
        else:
            # During inference (autoencoding only)
            seq = query_feats + latent_expanded

        # Add time embedding
        seq = seq + t_emb.unsqueeze(1)

        # Multi-directional MAMBA decoding
        for block in self.decoder_blocks:
            seq = block(seq, query_coords)  # 4-way spatial processing

        # Predict RGB
        pred_values = self.rgb_head(seq)  # (B, N_out, 3)

        return pred_values

    def forward(self, noisy_values, query_coords, t, input_coords, input_values):
        """
        Full forward pass for diffusion training

        This is the same signature as original MAMBADiffusion,
        so training code doesn't need to change!
        """
        # Encode semantic latent
        latent = self.encode(input_coords, input_values)

        # Decode with diffusion
        pred_velocity = self.decode(query_coords, latent, t, noisy_values)

        return pred_velocity
```

---

## Training Strategy

### Phase 1: Train Autoencoder (No Diffusion)

**Goal**: Learn semantic latent space for sparse fields

```python
def train_autoencoder(model, train_loader, epochs=100, lr=1e-4):
    """Train encoder-decoder without diffusion"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in train_loader:
            input_coords, input_values = batch['input_coords'], batch['input_values']
            output_coords, output_values = batch['output_coords'], batch['output_values']

            # Encode semantic latent
            latent = model.encode(input_coords, input_values)

            # Decode (no diffusion, just reconstruction)
            pred_values = model.decode(
                output_coords,
                latent,
                t=torch.zeros(B),  # no time, just reconstruction
                noisy_values=None
            )

            # MSE loss on reconstruction
            loss = F.mse_loss(pred_values, output_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate reconstruction quality
        if epoch % 10 == 0:
            eval_reconstruction(model, test_loader)
```

**Expected Outcome**:
- High-fidelity reconstruction (PSNR ~28-32 dB)
- Semantic latent space learned
- Multi-scale consistency improved (latent captures global structure)

---

### Phase 2: Train with Diffusion

**Goal**: Add stochastic details via diffusion process

```python
def train_diffusion(model, train_loader, epochs=100, lr=1e-4):
    """Train with flow matching diffusion"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in train_loader:
            # ... same as before ...

            # Encode semantic latent (FIXED during this phase)
            with torch.no_grad():
                latent = model.encode(input_coords, input_values)

            # Flow matching diffusion
            t = torch.rand(B)
            x_0 = torch.randn_like(output_values)
            x_1 = output_values
            x_t = (1 - t) * x_0 + t * x_1
            u_t = x_1 - x_0  # target velocity

            # Predict velocity with latent conditioning
            v_pred = model.decode(output_coords, latent, t, x_t)

            # Loss
            loss = F.mse_loss(v_pred, u_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**Key Difference from Standard MAMBA**:
- Standard: Diffuse entire reconstruction process
- DiffAE-MAMBA: Diffuse only stochastic details, semantic structure is deterministic

---

## Expected Benefits

### 1. Multi-Scale Consistency

**Problem**: Current MAMBA struggles at 64×64, 96×96
**Root Cause**: No global structure representation

**Solution**: Semantic latent captures global structure
```python
# Same latent → consistent reconstruction at all scales
latent = model.encode(coords_32x32, values_32x32)

# Decode at multiple scales
recon_32 = model.decode(query_coords_32, latent, t=0)
recon_64 = model.decode(query_coords_64, latent, t=0)
recon_96 = model.decode(query_coords_96, latent, t=0)

# Hypothesis: recon_64, recon_96 should be much smoother
# because they share the same semantic latent
```

---

### 2. Semantic Smoothness

**Problem**: Noise in reconstructions
**Root Cause**: Diffusing RGB values directly → high variance

**Solution**: Latent space is smoother
```python
# RGB space (noisy)
rgb_interp = 0.5 * rgb_1 + 0.5 * rgb_2  # pixel blending, jagged

# Latent space (smooth)
latent_1 = model.encode(coords_1, values_1)
latent_2 = model.encode(coords_2, values_2)
latent_interp = 0.5 * latent_1 + 0.5 * latent_2
rgb_interp = model.decode(query_coords, latent_interp, t=0)
# → smoother interpolation
```

---

### 3. Compression and Efficiency

**Current**: Diffuse 3 RGB values per coordinate
**Proposed**: Encode to 512-D latent, diffuse in latent space

**Latent Diffusion Variant**:
```python
# Option: Diffuse latent instead of RGB
def train_latent_diffusion(model, train_loader):
    for batch in train_loader:
        # Encode to latent
        latent_real = model.encode(input_coords, input_values)

        # Diffuse latent (512-D) instead of RGB (3×N)
        t = torch.rand(B)
        latent_noise = torch.randn_like(latent_real)
        latent_t = (1 - t) * latent_noise + t * latent_real

        # Predict velocity in latent space
        v_pred = latent_diffusion_net(latent_t, t)
        v_target = latent_real - latent_noise

        loss = F.mse_loss(v_pred, v_target)
```

**Benefits**:
- Much smaller diffusion space (512-D vs 3×4096 for 64×64)
- Faster convergence
- Better multi-scale generalization

---

## Testing Plan

### Experiment 1: Autoencoder Only (No Diffusion)

**Goal**: Test if semantic latent improves multi-scale consistency

```python
# Train autoencoder (Phase 1 only)
model = MAMBADiffusionAutoencoder(...)
train_autoencoder(model, train_loader, epochs=100)

# Evaluate reconstruction at multiple scales
for scale in [32, 64, 96]:
    coords, values = make_grid(scale)
    latent = model.encode(coords_sparse, values_sparse)
    recon = model.decode(coords_all, latent, t=0)

    psnr = compute_psnr(recon, ground_truth)
    print(f"{scale}×{scale}: PSNR = {psnr:.2f} dB")

# Compare with baseline MAMBA
# Hypothesis: DiffAE-MAMBA has better PSNR at 64×64, 96×96
```

**Success Criteria**:
- PSNR at 32×32: Similar to baseline (~24 dB)
- PSNR at 64×64: +2-4 dB improvement (smoother due to latent)
- PSNR at 96×96: +3-5 dB improvement
- Visual: Much smoother, less noisy

---

### Experiment 2: Autoencoder + Diffusion

**Goal**: Test if diffusion improves reconstruction quality

```python
# Fine-tune with diffusion (Phase 2)
train_diffusion(model, train_loader, epochs=100)

# Evaluate with diffusion sampling
for scale in [32, 64, 96]:
    coords, values = make_grid(scale)
    latent = model.encode(coords_sparse, values_sparse)

    # Sample with diffusion (Heun ODE solver)
    recon = heun_sample(
        model.decode,
        latent=latent,
        query_coords=coords_all,
        T=20  # DDIM-style fast sampling
    )

    psnr = compute_psnr(recon, ground_truth)
```

**Success Criteria**:
- Better PSNR than autoencoder-only
- Sharper edges, finer details
- Still smooth at large scales

---

### Experiment 3: Latent Interpolation

**Goal**: Test semantic smoothness

```python
# Encode two images
latent_1 = model.encode(coords_1, values_1)
latent_2 = model.encode(coords_2, values_2)

# Interpolate in latent space
for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    latent_interp = (1 - alpha) * latent_1 + alpha * latent_2
    recon = model.decode(coords_all, latent_interp, t=0)
    display(recon)

# Compare with RGB interpolation
rgb_interp = (1 - alpha) * rgb_1 + alpha * rgb_2

# Hypothesis: Latent interpolation is much smoother
```

---

## Implementation Roadmap

### Week 1: Autoencoder Implementation

```python
# Step 1: Implement MAMBADiffusionAutoencoder
# Step 2: Add encoder (multi-directional MAMBA + pooling)
# Step 3: Add decoder (latent conditioning)
# Step 4: Train on CIFAR-10 (autoencoder only)
# Step 5: Evaluate reconstruction quality
```

**Deliverable**: Working autoencoder with multi-scale evaluation

---

### Week 2: Diffusion Integration

```python
# Step 1: Add diffusion training loop
# Step 2: Integrate with existing flow matching code
# Step 3: Train on CIFAR-10 (autoencoder + diffusion)
# Step 4: Compare with baseline MAMBA
```

**Deliverable**: Full DiffAE-MAMBA model with quantitative comparison

---

### Week 3: Latent Diffusion Variant

```python
# Step 1: Implement latent diffusion network (MLP)
# Step 2: Train diffusion in latent space
# Step 3: Evaluate unconditional sampling
# Step 4: Compare latent vs RGB diffusion
```

**Deliverable**: Latent diffusion variant with ablation study

---

## Code Skeleton

**File**: `ASF/MAMBA_GINR_standalone/mamba_diffae.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_directional_mamba import MultiDirectionalMambaBlock
from core.neural_fields.perceiver import FourierFeatures


class MAMBADiffusionAutoencoder(nn.Module):
    """Semantic autoencoder for sparse fields"""
    def __init__(
        self,
        num_fourier_feats=256,
        d_model=512,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_state=16,
        dropout=0.1
    ):
        super().__init__()
        # ... (implementation from above) ...

    def encode(self, input_coords, input_values):
        # ... (implementation from above) ...

    def decode(self, query_coords, latent, t, noisy_values=None):
        # ... (implementation from above) ...

    def forward(self, noisy_values, query_coords, t, input_coords, input_values):
        # ... (implementation from above) ...


def train_autoencoder(model, train_loader, epochs=100, lr=1e-4, device='cuda'):
    """Phase 1: Train autoencoder (no diffusion)"""
    # ... (implementation from above) ...


def train_diffusion(model, train_loader, epochs=100, lr=1e-4, device='cuda'):
    """Phase 2: Train with diffusion"""
    # ... (implementation from above) ...
```

---

## Minimal Integration (Quick Test)

**If full DiffAE is too complex, try this minimal version first**:

```python
class MinimalLatentMAMBA(nn.Module):
    """
    Simplified version: Just add latent bottleneck to current MAMBA
    """
    def __init__(self):
        super().__init__()
        # Current MAMBA components
        self.input_proj = nn.Linear(feat_dim + 3, 512)
        self.mamba_blocks = nn.ModuleList([...])

        # NEW: Latent bottleneck
        self.to_latent = nn.Linear(512, 256)  # compress
        self.from_latent = nn.Linear(256, 512)  # expand

        self.decoder = nn.Linear(512, 3)

    def forward(self, noisy_values, query_coords, t, input_coords, input_values):
        # Encode
        seq = self.input_proj(torch.cat([feats, values], -1))
        for block in self.mamba_blocks:
            seq = block(seq, coords)

        # NEW: Latent bottleneck
        latent = self.to_latent(seq)  # (B, N, 256) - compressed
        seq = self.from_latent(latent)  # (B, N, 512) - expanded

        # Decode
        return self.decoder(seq)
```

**Test**: Does latent bottleneck improve multi-scale consistency?
- If yes → full DiffAE is promising
- If no → issue is elsewhere (serialization, training, etc.)

---

## Summary

### DiffAE Core Idea
Semantic encoder + conditional diffusion decoder = meaningful latent space

### Integration with MAMBA
1. **Autoencoder**: Multi-directional MAMBA encoder → latent → decoder
2. **Diffusion**: Diffuse in latent space, not RGB space
3. **Benefits**: Multi-scale consistency, semantic smoothness, compression

### Next Steps
1. ✅ Analyze DiffAE architecture (DONE)
2. ⏳ Implement MAMBADiffusionAutoencoder
3. ⏳ Train Phase 1 (autoencoder only)
4. ⏳ Evaluate multi-scale consistency
5. ⏳ Train Phase 2 (with diffusion)
6. ⏳ Compare with baseline MAMBA

### Expected Outcome
If DiffAE principles work for sparse fields:
- **PSNR improvement**: +2-4 dB at 64×64, +3-5 dB at 96×96
- **Visual quality**: Much smoother, less noisy
- **Multi-scale**: Consistent across scales

If not, then the issue is fundamental to MAMBA's sequential processing, and multi-directional scanning is the right solution.

---

**Questions for Discussion**:
1. Should we try minimal latent bottleneck first, or full DiffAE?
2. Which phase is more important: autoencoder or diffusion?
3. Should we diffuse in latent space or RGB space?

**Recommendation**: Start with autoencoder-only (Phase 1) to test semantic latent hypothesis. If it works, add diffusion (Phase 2).
