# Sparse Sampling Strategy

## Overview

The MAMBA diffusion model is trained with **instance-specific, deterministic, non-overlapping sparse sampling**.

## Key Properties

### 1. Instance-Specific Masking

Each image in the dataset has a **fixed, unique masking pattern** determined by:
- Dataset index (which image)
- Random seed (42 by default)

**Important**: The same image will **always** get the same input/output masks across:
- Multiple epochs
- Different training runs (with same seed)
- Multiple accesses to the same index

### 2. Non-Overlapping Input/Output

For each 32×32 CIFAR-10 image (1024 total pixels):

```
Total pixels: 1024 (100%)
  ├─ Input pixels:  205 (20%)  ─┐
  ├─ Output pixels: 205 (20%)  ─┼─ Non-overlapping!
  └─ Unused pixels: 614 (60%)  ─┘
```

**Guarantees**:
- Input and output pixel sets are **disjoint** (no overlap)
- Same 20% input pixels used throughout training
- Same 20% output pixels used as supervision throughout training

### 3. Sampling Process

```python
# Pre-computed at dataset initialization
for each image in dataset:
    # 1. Random permutation of all 1024 pixel indices
    perm = random.permutation([0, 1, 2, ..., 1023])

    # 2. First 20% → input pixels
    input_indices = perm[0:205]

    # 3. Next 20% → output pixels (guaranteed non-overlapping)
    output_indices = perm[205:410]

    # 4. Remaining 60% → unused
    unused_indices = perm[410:1024]
```

**Visual Example**:

```
Full Image (32×32)
┌────────────────────┐
│ ░░░░░▓▓▓░░░▓░░░░░ │  ░ = Input pixels (20%)
│ ░▓▓░░░░░▓▓░░░░░▓░ │  ▓ = Output pixels (20%)
│ ░░▓░░▓░░░░▓░░░░░░ │  (blank) = Unused (60%)
│ ░░░▓░░░░░░░░▓░░░░ │
│ ...               │
└────────────────────┘
```

## Why This Design?

### 1. Consistency in Training

**Problem**: Random sampling at each epoch means:
- Same image gets different masks every time
- Model can't learn image-specific patterns
- Unstable gradients across epochs

**Solution**: Fixed masks per image
- Model learns consistent representations
- Stable training dynamics
- Can track per-image reconstruction quality

### 2. Non-Overlapping Ensures Clean Supervision

**Problem**: Overlapping input/output
- Model could just copy input to output
- No generalization required
- "Cheating" on easy pixels

**Solution**: Disjoint sets force generalization
- Must interpolate/infer missing pixels
- Can't memorize direct mappings
- Learns true spatial understanding

### 3. Reproducibility

**Problem**: Different runs produce different results
- Hard to debug
- Can't compare across experiments
- Results not reproducible

**Solution**: Seeded sampling
- Same seed = same masks = same training
- Reproducible results
- Fair comparison across models

## Code Implementation

### Dataset Creation

```python
from core.sparse.cifar10_sparse import SparseCIFAR10Dataset

# Create dataset with deterministic sampling
dataset = SparseCIFAR10Dataset(
    root='./data',
    train=True,
    input_ratio=0.2,   # 20% input
    output_ratio=0.2,  # 20% output (non-overlapping)
    seed=42            # Fixed seed for reproducibility
)
```

### Pre-Generation at Init

Sampling indices are **pre-generated once** at dataset initialization:

```python
# From cifar10_sparse.py line 69-89
def _generate_sampling_indices(self):
    """Pre-generate random sampling indices for all instances"""
    for i in range(len(dataset)):
        # Random permutation (deterministic with seed)
        perm = self.rng.permutation(total_pixels)

        # First 20% → input
        input_idx = perm[:n_input_pixels]

        # Next 20% → output (guaranteed disjoint)
        output_idx = perm[n_input_pixels:n_input_pixels + n_output_pixels]

        # Store for fast lookup
        self.input_indices.append(input_idx)
        self.output_indices.append(output_idx)
```

### Fast Retrieval

```python
# From cifar10_sparse.py line 115-153
def __getitem__(self, idx):
    # Get pre-computed indices (O(1) lookup)
    input_idx = self.input_indices[idx]
    output_idx = self.output_indices[idx]

    # Extract pixel values
    input_values = image[:, input_idx]
    output_values = image[:, output_idx]

    return {
        'input_coords': coords(input_idx),
        'input_values': input_values,
        'output_coords': coords(output_idx),
        'output_values': output_values,
        ...
    }
```

## Training Flow

### Forward Pass

```python
# Training loop (train_mamba_standalone.py lines 375-392)
for batch in train_loader:
    # 1. Get sparse observations (20% pixels)
    input_coords = batch['input_coords']  # (B, 205, 2)
    input_values = batch['input_values']  # (B, 205, 3)

    # 2. Get target output locations (different 20%)
    output_coords = batch['output_coords']  # (B, 205, 2)
    output_values = batch['output_values']  # (B, 205, 3)

    # 3. Flow matching training
    t = random([0, 1])  # Timestep
    x_t = interpolate(noise, output_values, t)

    # 4. Predict velocity conditioned on input
    velocity = model(x_t, output_coords, t,
                     input_coords, input_values)

    # 5. Loss: predicted velocity should match target
    loss = MSE(velocity, output_values - noise)
```

### Key Point

The model receives:
- **Context**: 20% observed pixels (input)
- **Query**: 20% different pixel locations (output coords)
- **Task**: Predict RGB values at query locations

This forces the model to:
1. Encode spatial patterns from sparse input
2. Generalize to unseen pixel locations
3. Interpolate smoothly between observations

## Verification

Run the verification script to confirm correct behavior:

```bash
cd ASF
python verify_deterministic_masking.py
```

This will test:
1. ✅ Deterministic sampling (same seed = same masks)
2. ✅ Non-overlapping input/output (disjoint sets)
3. ✅ Consistency (same instance = same masks)
4. ✅ Coverage (exactly 40% of pixels used)

Output:
```
TEST 1: Deterministic Sampling
✅ PASS: All instances have deterministic sampling

TEST 2: Non-Overlapping Input/Output
✅ PASS: All instances have non-overlapping input/output

TEST 3: Consistency Across Multiple Accesses
✅ PASS: Instance returns consistent masks across accesses

TEST 4: Pixel Coverage
✅ PASS: All instances have ~40% coverage

🎉 All tests passed!
```

## Advantages Over Alternatives

### vs. Random Sampling Every Epoch

| Property | Our Approach | Random Each Epoch |
|----------|--------------|-------------------|
| Consistency | ✅ Fixed masks | ❌ Different every time |
| Stability | ✅ Stable gradients | ❌ Noisy gradients |
| Reproducibility | ✅ Deterministic | ❌ Non-reproducible |
| Training speed | ✅ Fast (pre-computed) | ⚠️ Slower (runtime sampling) |

### vs. Grid Sampling

| Property | Our Approach | Grid Pattern |
|----------|--------------|--------------|
| Spatial coverage | ✅ Uniform random | ⚠️ Regular pattern |
| Generalization | ✅ Diverse patterns | ❌ Overfits to grid |
| Realism | ✅ Natural sparsity | ❌ Artificial structure |

### vs. Overlapping Input/Output

| Property | Our Approach | Overlapping |
|----------|--------------|-------------|
| Task difficulty | ✅ Must interpolate | ❌ Can copy |
| Generalization | ✅ True understanding | ❌ Memorization |
| Evaluation | ✅ Fair test | ❌ Inflated metrics |

## Summary

✅ **Instance-specific**: Same image always gets same masks
✅ **Deterministic**: Seeded RNG for reproducibility
✅ **Non-overlapping**: Input and output are disjoint sets
✅ **Pre-computed**: Fast O(1) lookup during training
✅ **Balanced**: Exactly 20% input + 20% output = 40% total

This design ensures **consistent, challenging, and reproducible** training for sparse-to-dense neural field reconstruction.
