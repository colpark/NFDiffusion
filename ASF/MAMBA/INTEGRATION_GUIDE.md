# Integration Guide: Multi-Directional MAMBA

**Goal**: Add multi-directional scanning to your existing MAMBA notebook to address spatial locality issues.

**Expected Result**: Smoother reconstructions with less noise/wiggliness.

---

## Quick Start (3 Steps)

### Step 1: Add Multi-Directional Code Cell

**Insert this cell AFTER your existing `SSMBlock` and `MambaBlock` definitions**:

```python
# ============================================
# MULTI-DIRECTIONAL MAMBA COMPONENTS
# ============================================

def order_by_row(coords):
    """Row-major ordering (horizontal scan)"""
    B, N, _ = coords.shape
    indices_list = []
    for b in range(B):
        y_vals = coords[b, :, 1]
        x_vals = coords[b, :, 0]
        sort_keys = y_vals * 1000 + x_vals
        indices = torch.argsort(sort_keys)
        indices_list.append(indices)
    return torch.stack(indices_list, dim=0)

def order_by_column(coords):
    """Column-major ordering (vertical scan)"""
    B, N, _ = coords.shape
    indices_list = []
    for b in range(B):
        y_vals = coords[b, :, 1]
        x_vals = coords[b, :, 0]
        sort_keys = x_vals * 1000 + y_vals
        indices = torch.argsort(sort_keys)
        indices_list.append(indices)
    return torch.stack(indices_list, dim=0)

def order_by_diagonal(coords):
    """Diagonal ordering (top-left to bottom-right)"""
    B, N, _ = coords.shape
    indices_list = []
    for b in range(B):
        y_vals = coords[b, :, 1]
        x_vals = coords[b, :, 0]
        diag_vals = x_vals + y_vals
        sort_keys = diag_vals * 1000 + x_vals
        indices = torch.argsort(sort_keys)
        indices_list.append(indices)
    return torch.stack(indices_list, dim=0)

def order_by_antidiagonal(coords):
    """Anti-diagonal ordering (top-right to bottom-left)"""
    B, N, _ = coords.shape
    indices_list = []
    for b in range(B):
        y_vals = coords[b, :, 1]
        x_vals = coords[b, :, 0]
        antidiag_vals = x_vals - y_vals
        sort_keys = antidiag_vals * 1000 + x_vals
        indices = torch.argsort(sort_keys)
        indices_list.append(indices)
    return torch.stack(indices_list, dim=0)

def reorder_sequence(x, indices):
    """Apply ordering to sequence"""
    B, N, D = x.shape
    indices_expanded = indices.unsqueeze(-1).expand(B, N, D)
    return torch.gather(x, dim=1, index=indices_expanded)

def inverse_reorder(x, indices):
    """Reverse ordering back to original positions"""
    B, N, D = x.shape
    inverse_indices = torch.zeros_like(indices)
    for b in range(B):
        inverse_indices[b, indices[b]] = torch.arange(N, device=indices.device)
    indices_expanded = inverse_indices.unsqueeze(-1).expand(B, N, D)
    return torch.gather(x, dim=1, index=indices_expanded)


class MultiDirectionalSSM(nn.Module):
    """Multi-directional State Space Model"""
    def __init__(self, ssm_block_class, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 4 separate SSM blocks for each direction
        self.ssm_horizontal = ssm_block_class(d_model, d_state, dropout)
        self.ssm_vertical = ssm_block_class(d_model, d_state, dropout)
        self.ssm_diagonal = ssm_block_class(d_model, d_state, dropout)
        self.ssm_antidiagonal = ssm_block_class(d_model, d_state, dropout)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(4 * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, coords):
        """
        Args:
            x: (B, N, d_model) features
            coords: (B, N, 2) spatial coordinates
        Returns:
            y: (B, N, d_model) with 2D spatial awareness
        """
        # Get orderings
        indices_h = order_by_row(coords)
        indices_v = order_by_column(coords)
        indices_d = order_by_diagonal(coords)
        indices_a = order_by_antidiagonal(coords)

        # Horizontal
        x_h = reorder_sequence(x, indices_h)
        y_h = self.ssm_horizontal(x_h)
        y_h = inverse_reorder(y_h, indices_h)

        # Vertical
        x_v = reorder_sequence(x, indices_v)
        y_v = self.ssm_vertical(x_v)
        y_v = inverse_reorder(y_v, indices_v)

        # Diagonal
        x_d = reorder_sequence(x, indices_d)
        y_d = self.ssm_diagonal(x_d)
        y_d = inverse_reorder(y_d, indices_d)

        # Anti-diagonal
        x_a = reorder_sequence(x, indices_a)
        y_a = self.ssm_antidiagonal(x_a)
        y_a = inverse_reorder(y_a, indices_a)

        # Fuse
        y_concat = torch.cat([y_h, y_v, y_d, y_a], dim=-1)
        y_fused = self.fusion(y_concat)

        # Residual
        y = x + y_fused
        y = self.norm(y)

        return y


class MultiDirectionalMambaBlock(nn.Module):
    """Complete multi-directional Mamba block"""
    def __init__(self, d_model, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()

        self.proj_in = nn.Linear(d_model, d_model * expand_factor)

        # Use SSMBlockFast (defined earlier in notebook)
        self.multi_ssm = MultiDirectionalSSM(
            SSMBlockFast,  # Reference to your existing SSMBlockFast class
            d_model * expand_factor,
            d_state,
            dropout
        )

        self.proj_out = nn.Linear(d_model * expand_factor, d_model)

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, coords):
        """
        Args:
            x: (B, N, d_model)
            coords: (B, N, 2) spatial coordinates
        """
        # SSM branch
        residual = x
        x = self.proj_in(x)
        x = self.multi_ssm(x, coords)
        x = self.proj_out(x)
        x = x + residual

        # MLP branch
        x = x + self.mlp(x)

        return x


print("✓ Multi-directional MAMBA components loaded")
print("  - 4 scanning directions: horizontal, vertical, diagonal, anti-diagonal")
print("  - Fusion mechanism to combine all directions")
print("  - Ready to integrate into MAMBADiffusion")
```

---

### Step 2: Replace MAMBADiffusion Class

**Find your existing `MAMBADiffusion` class and replace it with this**:

```python
class MAMBADiffusion(nn.Module):
    """
    Multi-Directional State Space Model for sparse field diffusion

    CHANGES from v1:
    - Uses MultiDirectionalMambaBlock instead of MambaBlock
    - Passes coordinates to MAMBA blocks for directional scanning
    - Captures full 2D spatial structure (horizontal, vertical, diagonal)
    """
    def __init__(
        self,
        num_fourier_feats=256,
        d_model=512,
        num_layers=6,
        d_state=16,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Fourier features
        self.fourier = FourierFeatures(coord_dim=2, num_freqs=num_fourier_feats, scale=10.0)
        feat_dim = num_fourier_feats * 2

        # Project inputs and queries
        self.input_proj = nn.Linear(feat_dim + 3, d_model)
        self.query_proj = nn.Linear(feat_dim + 3, d_model)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # ============================================
        # CHANGED: Multi-directional MAMBA blocks
        # ============================================
        self.mamba_blocks = nn.ModuleList([
            MultiDirectionalMambaBlock(
                d_model,
                d_state=d_state,
                expand_factor=2,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Cross-attention to extract query-specific features
        self.query_cross_attn = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=dropout, batch_first=True
        )

        # Output decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )

    def forward(self, noisy_values, query_coords, t, input_coords, input_values):
        """
        Args:
            noisy_values: (B, N_out, 3)
            query_coords: (B, N_out, 2)
            t: (B,) timestep
            input_coords: (B, N_in, 2)
            input_values: (B, N_in, 3)
        """
        B = query_coords.shape[0]
        N_in = input_coords.shape[1]
        N_out = query_coords.shape[1]

        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))  # (B, d_model)

        # Fourier features
        input_feats = self.fourier(input_coords)  # (B, N_in, feat_dim)
        query_feats = self.fourier(query_coords)  # (B, N_out, feat_dim)

        # Encode inputs and queries
        input_tokens = self.input_proj(
            torch.cat([input_feats, input_values], dim=-1)
        )
        query_tokens = self.query_proj(
            torch.cat([query_feats, noisy_values], dim=-1)
        )

        # Add time embedding
        input_tokens = input_tokens + t_emb.unsqueeze(1)
        query_tokens = query_tokens + t_emb.unsqueeze(1)

        # ============================================
        # CHANGED: Concatenate coordinates and pass to MAMBA
        # ============================================
        all_coords = torch.cat([input_coords, query_coords], dim=1)  # (B, N_total, 2)
        seq = torch.cat([input_tokens, query_tokens], dim=1)  # (B, N_total, d_model)

        # Process through multi-directional MAMBA blocks
        for mamba_block in self.mamba_blocks:
            seq = mamba_block(seq, all_coords)  # ← Pass coords!

        # Split back
        input_seq = seq[:, :N_in, :]
        query_seq = seq[:, N_in:, :]

        # Cross-attention
        output, _ = self.query_cross_attn(query_seq, input_seq, input_seq)

        # Decode to RGB
        return self.decoder(output)


print("✓ Multi-directional MAMBADiffusion model ready")
print("  - Replaces standard MAMBA with multi-directional version")
print("  - All 4 directions processed and fused")
print("  - API unchanged - drop-in replacement for training code")
```

---

### Step 3: Test the Model (Quick Validation)

**Add this cell to test the new architecture**:

```python
# Test multi-directional model
print("Testing Multi-Directional MAMBA...")
print("=" * 60)

model_multidir = MAMBADiffusion(
    num_fourier_feats=256,
    d_model=512,
    num_layers=6,
    d_state=16
).to(device)

# Test forward pass
test_noisy = torch.rand(4, 204, 3).to(device)
test_query_coords = torch.rand(4, 204, 2).to(device)
test_t = torch.rand(4).to(device)
test_input_coords = torch.rand(4, 204, 2).to(device)
test_input_values = torch.rand(4, 204, 3).to(device)

test_out = model_multidir(test_noisy, test_query_coords, test_t, test_input_coords, test_input_values)

print(f"✓ Forward pass successful!")
print(f"  Input shape: {test_noisy.shape}")
print(f"  Output shape: {test_out.shape}")
print(f"  Output range: [{test_out.min():.3f}, {test_out.max():.3f}]")

total_params = sum(p.numel() for p in model_multidir.parameters())
print(f"\n✓ Model parameters: {total_params:,}")
print(f"  Expected: ~4× more than v1 (due to 4 SSM blocks per layer)")

print("\n" + "=" * 60)
print("MODEL READY TO TRAIN!")
print("=" * 60)
print("\nNo changes needed to training code - just run:")
print("  losses = train_flow_matching(model_multidir, ...)")
```

---

## Training the Multi-Directional Model

### Option 1: Quick Test (1-5 epochs)

```python
# Quick test to see if noise reduces
model_multidir = MAMBADiffusion(
    num_fourier_feats=256,
    d_model=512,
    num_layers=6,
    d_state=16
).to(device)

# Train for just a few epochs to validate
losses = train_flow_matching(
    model_multidir,
    train_loader,
    test_loader,
    epochs=5,
    lr=1e-4,
    device=device,
    save_dir='checkpoints_multidir_test'
)

# Compare visual quality with v1
# → If significantly smoother → hypothesis confirmed!
```

### Option 2: Short Training (20 epochs)

```python
# If quick test shows promise, train for 20 epochs
losses = train_flow_matching(
    model_multidir,
    train_loader,
    test_loader,
    epochs=20,
    lr=1e-4,
    device=device,
    save_dir='checkpoints_multidir_v1'
)

# Evaluate multi-scale quality
# Run your multi-scale evaluation code
```

### Option 3: Full Training (100-200 epochs)

```python
# Full training with proper LR schedule
model_multidir = MAMBADiffusion(...).to(device)

# Better LR schedule (from training duration analysis)
warmup_epochs = 20

def get_lr_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (200 - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

optimizer = torch.optim.Adam(model_multidir.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

# Train fully
losses = train_flow_matching(
    model_multidir,
    train_loader,
    test_loader,
    epochs=200,
    lr=1e-4,
    device=device,
    save_dir='checkpoints_multidir_full'
)
```

---

## Expected Results

### Visual Improvements

**If multi-directional scanning solves the spatial locality problem:**

✅ **32×32**: Smooth textures, clear edges, no pixel-level noise
✅ **64×64**: Natural-looking upsampling, better than bilinear
✅ **96×96**: Coherent structures, consistent colors

**Before (v1 - Single Direction)**:
```
[Noisy, wiggly textures]
[Pixel-level artifacts]
[Poor multi-scale quality]
```

**After (Multi-Directional)**:
```
[Smooth gradients]
[Clean textures]
[Natural multi-scale]
```

### Quantitative Improvements

| Metric | v1 (Single) | Multi-Dir | Improvement |
|--------|-------------|-----------|-------------|
| PSNR (32×32) | ~24 dB | 26-28 dB | +2-4 dB |
| SSIM (32×32) | ~0.85 | 0.90-0.92 | +0.05-0.07 |
| Visual Quality | Noisy | Smooth | Significant |

---

## Troubleshooting

### Issue 1: Out of Memory

**Problem**: 4× more SSM parameters → OOM

**Solutions**:
1. Reduce `d_state` from 16 to 8
2. Reduce `num_layers` from 6 to 4
3. Reduce `batch_size` from 64 to 32
4. Use gradient checkpointing

```python
# Smaller model
model_multidir = MAMBADiffusion(
    d_model=512,
    num_layers=4,  # ← Reduced
    d_state=8,      # ← Reduced
).to(device)
```

### Issue 2: Very Slow Training

**Problem**: 4× SSM blocks → 3-4× slower

**Solutions**:
1. Train fewer epochs to start (20 instead of 200)
2. Use lightweight version (shared weights - see below)
3. Reduce `num_steps` in heun_sample during training visualization

### Issue 3: No Quality Improvement

**Problem**: Multi-directional doesn't help noise

**Interpretation**:
- Spatial locality is NOT the main issue
- Need to investigate other causes:
  - Training duration (200 epochs insufficient)
  - Fourier scale (scale=10 too high)
  - ODE steps (50 too few)

**Next Steps**: Try the other fixes from PROBLEMS_ANALYSIS.md

---

## Lightweight Alternative (If Memory/Speed is Issue)

**Shared-weight version** (same parameters as v1, but still multi-directional):

```python
class LightweightMultiDirectionalBlock(nn.Module):
    """Multi-directional with shared SSM weights"""
    def __init__(self, d_model, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()

        self.proj_in = nn.Linear(d_model, d_model * expand_factor)

        # SINGLE SSM for all directions
        self.ssm = SSMBlockFast(d_model * expand_factor, d_state, dropout)

        # Lightweight fusion
        self.fusion = nn.Linear(d_model * expand_factor * 4, d_model * expand_factor)

        self.proj_out = nn.Linear(d_model * expand_factor, d_model)

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, coords):
        residual = x
        x = self.proj_in(x)

        # Get 4 orderings
        indices = [
            order_by_row(coords),
            order_by_column(coords),
            order_by_diagonal(coords),
            order_by_antidiagonal(coords)
        ]

        # Process each direction with SAME SSM
        outputs = []
        for idx in indices:
            x_ordered = reorder_sequence(x, idx)
            y = self.ssm(x_ordered)
            y = inverse_reorder(y, idx)
            outputs.append(y)

        # Fuse
        y_fused = self.fusion(torch.cat(outputs, dim=-1))

        x = self.proj_out(y_fused)
        x = x + residual
        x = x + self.mlp(x)

        return x
```

**Use this in MAMBADiffusion by replacing `MultiDirectionalMambaBlock` with `LightweightMultiDirectionalBlock`**

---

## Summary

1. **Add** multi-directional components cell
2. **Replace** MAMBADiffusion class
3. **Test** with 1-5 epochs first
4. **If successful** → train fully (100-200 epochs)
5. **Compare** visual quality with v1

**Expected**: If spatial locality is the issue → significant noise reduction!

If not → move to next hypothesis (training duration, Fourier scale, etc.)
