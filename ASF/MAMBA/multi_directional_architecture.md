# Multi-Directional MAMBA Architecture

## Problem Statement

**Root Cause of Noise**: MAMBA processes coordinates as a 1D sequence, losing 2D spatial structure.

```
2D Image Grid:              MAMBA's Sequential View:
┌───┬───┬───┬───┐
│ A │ B │ C │ D │           A → B → C → D → E → F → G → H → ...
├───┼───┼───┼───┤                   ↑
│ E │ F │ G │ H │           Only sees left neighbor,
└───┴───┴───┴───┘           NOT spatial neighbors (above/below)!
```

**Result**: Model can't capture 2D textures and local structure → **noise and wiggliness**

---

## Solution: Multi-Directional Scanning

Process the same coordinates in **4 different orderings**, then **fuse** the results:

1. **Horizontal** (row-wise): `A→B→C→D, E→F→G→H, ...`
2. **Vertical** (column-wise): `A→E→..., B→F→..., ...`
3. **Diagonal**: `A→B→E→C→F→..., ...`
4. **Anti-diagonal**: `D→C→H→..., ...`

Each direction captures different spatial relationships. Fusion combines all perspectives.

---

## Architecture Changes

### Before (Single-Path MAMBA)

```python
class MAMBADiffusion(nn.Module):
    def __init__(self, ...):
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state)  # Single direction
            for _ in range(num_layers)
        ])

    def forward(self, noisy_values, query_coords, t, input_coords, input_values):
        # ... encoding ...
        seq = torch.cat([input_tokens, query_tokens], dim=1)

        # Process through MAMBA (single ordering)
        for mamba_block in self.mamba_blocks:
            seq = mamba_block(seq)  # ← NO spatial awareness!

        # ... decoding ...
```

### After (Multi-Directional MAMBA)

```python
from multi_directional_mamba import MultiDirectionalMambaBlock, SSMBlockFast

class MultiDirectionalMAMBADiffusion(nn.Module):
    def __init__(self, ...):
        self.mamba_blocks = nn.ModuleList([
            MultiDirectionalMambaBlock(  # ← Multi-directional
                SSMBlockFast,
                d_model,
                d_state,
                expand_factor=2,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, noisy_values, query_coords, t, input_coords, input_values):
        # ... encoding ...
        seq = torch.cat([input_tokens, query_tokens], dim=1)

        # Coordinates for ordering
        coords = torch.cat([input_coords, query_coords], dim=1)

        # Process through multi-directional MAMBA
        for mamba_block in self.mamba_blocks:
            seq = mamba_block(seq, coords)  # ← 4 directions + fusion!

        # ... decoding ...
```

---

## Data Flow

```
Input:
├─ seq: [B, N_total, d_model]  (features)
└─ coords: [B, N_total, 2]     (spatial positions)

                ↓

Multi-Directional Processing:

  ┌────────────────────────────────────────────┐
  │  Direction 1: Horizontal                   │
  │  ────────────────────────                  │
  │  coords → sort by (y, x)                   │
  │         → reorder seq                      │
  │         → SSM_horizontal(seq)              │
  │         → reverse reorder → y_h            │
  └────────────────────────────────────────────┘

  ┌────────────────────────────────────────────┐
  │  Direction 2: Vertical                     │
  │  ────────────────────────────              │
  │  coords → sort by (x, y)                   │
  │         → reorder seq                      │
  │         → SSM_vertical(seq)                │
  │         → reverse reorder → y_v            │
  └────────────────────────────────────────────┘

  ┌────────────────────────────────────────────┐
  │  Direction 3: Diagonal                     │
  │  ────────────────────────────              │
  │  coords → sort by (x+y, x)                 │
  │         → reorder seq                      │
  │         → SSM_diagonal(seq)                │
  │         → reverse reorder → y_d            │
  └────────────────────────────────────────────┘

  ┌────────────────────────────────────────────┐
  │  Direction 4: Anti-diagonal                │
  │  ────────────────────────────              │
  │  coords → sort by (x-y, x)                 │
  │         → reorder seq                      │
  │         → SSM_antidiagonal(seq)            │
  │         → reverse reorder → y_a            │
  └────────────────────────────────────────────┘

                ↓

Fusion:
  [y_h | y_v | y_d | y_a] → Linear(4*d → 2*d) → GELU → Linear(2*d → d)

  Result: y_fused [B, N_total, d_model]

                ↓

Output: seq_out = seq + y_fused  (residual connection)
```

---

## Key Implementation Details

### 1. Ordering Functions

```python
def order_by_row(coords):
    """Sort by (y, x) for horizontal scanning"""
    y, x = coords[..., 1], coords[..., 0]
    sort_keys = y * 1000 + x  # Composite key
    indices = torch.argsort(sort_keys)
    return indices

def order_by_column(coords):
    """Sort by (x, y) for vertical scanning"""
    # Similar implementation

def order_by_diagonal(coords):
    """Sort by (x+y, x) for diagonal scanning"""
    # Similar implementation

def order_by_antidiagonal(coords):
    """Sort by (x-y, x) for anti-diagonal scanning"""
    # Similar implementation
```

### 2. Reordering Operations

```python
def reorder_sequence(x, indices):
    """Apply ordering to sequence"""
    B, N, D = x.shape
    indices_expanded = indices.unsqueeze(-1).expand(B, N, D)
    return torch.gather(x, dim=1, index=indices_expanded)

def inverse_reorder(x, indices):
    """Reverse ordering back to original positions"""
    # Create inverse permutation
    inverse_indices = torch.zeros_like(indices)
    for b in range(B):
        inverse_indices[b, indices[b]] = torch.arange(N)
    return reorder_sequence(x, inverse_indices)
```

### 3. Fusion Mechanism

Two strategies implemented:

**Strategy 1: Concatenate + Project** (default)
```python
y_concat = torch.cat([y_h, y_v, y_d, y_a], dim=-1)  # (B, N, 4*d_model)
y_fused = self.fusion(y_concat)  # MLP: 4*d → 2*d → d
```

**Strategy 2: Learnable Weighted Average** (alternative)
```python
weights = F.softmax(self.direction_weights, dim=0)
y_fused = weights[0]*y_h + weights[1]*y_v + weights[2]*y_d + weights[3]*y_a
```

---

## Integration Guide

### Step 1: Import Multi-Directional Modules

Add to your notebook:

```python
import sys
sys.path.insert(0, 'MAMBA/')
from multi_directional_mamba import (
    MultiDirectionalMambaBlock,
    MultiDirectionalSSM,
    order_by_row,
    order_by_column,
    order_by_diagonal,
    order_by_antidiagonal
)
```

### Step 2: Replace MambaBlock with MultiDirectionalMambaBlock

In `MAMBADiffusion.__init__()`:

```python
# OLD:
self.mamba_blocks = nn.ModuleList([
    MambaBlock(d_model, d_state, expand_factor=2, dropout=dropout)
    for _ in range(num_layers)
])

# NEW:
self.mamba_blocks = nn.ModuleList([
    MultiDirectionalMambaBlock(
        SSMBlockFast,  # Pass the SSM class
        d_model,
        d_state,
        expand_factor=2,
        dropout=dropout
    )
    for _ in range(num_layers)
])
```

### Step 3: Pass Coordinates to MAMBA Blocks

In `MAMBADiffusion.forward()`:

```python
# Concatenate coords BEFORE processing
all_coords = torch.cat([input_coords, query_coords], dim=1)  # (B, N_total, 2)
seq = torch.cat([input_tokens, query_tokens], dim=1)  # (B, N_total, d_model)

# OLD:
for mamba_block in self.mamba_blocks:
    seq = mamba_block(seq)

# NEW:
for mamba_block in self.mamba_blocks:
    seq = mamba_block(seq, all_coords)  # ← Pass coords!
```

---

## Expected Improvements

### Hypothesis
**If noise is caused by missing 2D spatial structure**, multi-directional MAMBA should:

✅ **Reduce high-frequency noise** (better local context)
✅ **Smoother textures** (captures 2D patterns)
✅ **Better edge preservation** (diagonal scans capture corners)
✅ **Improved multi-scale generalization** (fundamental improvement)

### Metrics to Watch

| Metric | Current (v1) | Expected (Multi-Dir) |
|--------|--------------|---------------------|
| **Training Loss** | Baseline | Similar or better |
| **Visual Smoothness** | Noisy/wiggly | Smooth textures |
| **PSNR (32×32)** | ~24 dB | 26-28 dB (+2-4 dB) |
| **SSIM (32×32)** | ~0.85 | 0.90-0.92 |
| **64×64 Quality** | Very noisy | Much smoother |
| **96×96 Quality** | Extremely noisy | Smoother |

### What Success Looks Like

**Visual Inspection**:
- Smooth color gradients (no pixel-level noise)
- Clear edges and textures
- Consistent patterns across scales
- Natural-looking reconstructions

**Failure Mode** (if multi-directional doesn't help):
- Similar noise levels → Problem is NOT serialization
- Need to investigate other causes (training duration, Fourier scale, etc.)

---

## Computational Cost

### Memory
- **4× more SSM blocks**: Each direction has separate parameters
- **Original**: `num_layers × d_model² × 2` (one SSM per layer)
- **Multi-dir**: `num_layers × d_model² × 2 × 4` (four SSMs per layer)
- **Increase**: ~**4× parameters** in SSM layers

### Speed
- **4× forward passes** through SSM per layer
- **Additional**: Reordering operations (cheap), fusion MLP
- **Estimated**: ~**3-4× slower** training

### Mitigation
- **Shared weights**: Make all 4 directions share SSM weights (reduces params to 1×)
- **Selective layers**: Only use multi-directional in bottom 3 layers
- **Smaller d_state**: Reduce from 16 to 8 to compensate

---

## Testing Plan

### Phase 1: Quick Validation (1 epoch)
```python
# Train for 1 epoch with multi-directional MAMBA
model = MultiDirectionalMAMBADiffusion(...)
losses = train_flow_matching(model, ..., epochs=1)

# Visualize: Compare with v1 visually
# → Should see immediate smoothness improvement if hypothesis is correct
```

### Phase 2: Short Training (20 epochs)
```python
# Train for 20 epochs
losses = train_flow_matching(model, ..., epochs=20)

# Evaluate:
# - Training loss trajectory
# - Visual quality at 32×32
# - Multi-scale quality (64×64, 96×96)
```

### Phase 3: Full Training (100-200 epochs)
```python
# If Phase 2 shows promise, train fully
losses = train_flow_matching(model, ..., epochs=200)

# Final evaluation:
# - Full metrics (PSNR, SSIM)
# - Multi-scale evaluation
# - Comparison with v1
```

---

## Ablation Studies (Optional)

### Test Individual Directions
Which direction contributes most?

```python
# Test fusion weights after training
print(model.mamba_blocks[0].multi_ssm.direction_weights)

# Expected: All ~0.25 (equal contribution)
# If one dominates: That direction is most important
```

### Test Fusion Strategies
- **Concat + Project** (current)
- **Weighted Average**
- **Attention-based fusion**

```python
# Switch fusion in MultiDirectionalSSM
# Compare: Training speed, final quality
```

---

## Alternative: Lighter-Weight Version

If 4× parameters is too expensive:

### Shared-Weight Multi-Directional MAMBA

```python
class LightweightMultiDirectionalSSM(nn.Module):
    def __init__(self, ssm_block_class, d_model, d_state=16, dropout=0.1):
        super().__init__()

        # SINGLE SSM shared across all directions
        self.ssm = ssm_block_class(d_model, d_state, dropout)

        # Lightweight fusion
        self.fusion = nn.Linear(d_model * 4, d_model)

    def forward(self, x, coords):
        # Get 4 orderings
        indices = [
            order_by_row(coords),
            order_by_column(coords),
            order_by_diagonal(coords),
            order_by_antidiagonal(coords)
        ]

        outputs = []
        for idx in indices:
            x_ordered = reorder_sequence(x, idx)
            y = self.ssm(x_ordered)  # ← SAME SSM
            y = inverse_reorder(y, idx)
            outputs.append(y)

        # Fuse
        y_fused = self.fusion(torch.cat(outputs, dim=-1))
        return x + y_fused
```

**Trade-off**:
- ✅ Same parameters as v1 (no increase!)
- ✅ Still gets multi-directional information
- ⚠️ Shared weights may learn "average" instead of direction-specific patterns

---

## Summary

**Problem**: MAMBA's 1D sequential processing loses 2D spatial structure
**Solution**: Process in 4 directions, fuse results
**Cost**: 4× more parameters, 3-4× slower training
**Expected**: 2-4 dB PSNR improvement, much smoother outputs
**Test**: Train 1-20 epochs first, check if noise reduces

If multi-directional MAMBA significantly reduces noise → confirms spatial locality hypothesis!
