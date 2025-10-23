# Multi-Scale Evaluation Guide

## Overview

This evaluation tests two critical properties of continuous implicit neural fields:

1. **Full Field Reconstruction**: Can the model reconstruct the ENTIRE image (all 1024 pixels), not just the 20% output pixels used in training?

2. **Scale Invariance**: Can the model generalize to higher resolutions (64x64, 96x96, 128x128) despite being trained only on 32x32?

## Why This Matters

### Traditional CNNs
- **Fixed resolution**: Trained on 32x32, can only output 32x32
- **Upsampling required**: Must use bilinear/bicubic interpolation for higher resolutions
- **Grid-dependent**: Tied to specific pixel grid

### Continuous Implicit Fields (Our Approach)
- **Resolution-free**: Trained on 32x32, but can query at ANY resolution
- **Native upsampling**: Directly sample at 64x64, 96x96, etc.
- **Grid-independent**: Fourier features enable continuous coordinate queries

## Key Hypothesis

**If the model learned truly continuous representations via Fourier features, it should:**
- Reconstruct the full 32x32 image with high quality
- Generalize to 64x64, 96x96, and beyond
- Produce sharper results than traditional bilinear upsampling
- Show no grid-aligned artifacts at higher resolutions

## Evaluation Script Usage

### Basic Usage

```bash
cd ASF

# For Local Implicit model
python evaluate_multiscale.py \
    --model local_implicit \
    --checkpoint local_implicit_trained.pth \
    --scales 32 64 96 128 \
    --output_dir ./results_local

# For MAMBA model
python evaluate_multiscale.py \
    --model mamba \
    --checkpoint mamba_trained.pth \
    --scales 32 64 96 128 \
    --output_dir ./results_mamba
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *required* | Model type: `local_implicit` or `mamba` |
| `--checkpoint` | *required* | Path to trained model checkpoint (.pth file) |
| `--batch_size` | 16 | Batch size for evaluation |
| `--num_steps` | 100 | ODE solver steps (higher = better quality) |
| `--output_dir` | `./multiscale_results` | Where to save visualizations |
| `--scales` | `[32, 64, 96]` | Resolutions to evaluate |

### Example with Custom Settings

```bash
python evaluate_multiscale.py \
    --model local_implicit \
    --checkpoint models/local_implicit_epoch_100.pth \
    --scales 32 48 64 96 128 \
    --num_steps 150 \
    --batch_size 8 \
    --output_dir ./high_quality_results
```

## Output Files

The script generates:

1. **Multi-scale visualizations**: `{model}_multiscale_sample_{i}.png`
   - Shows ground truth, sparse input, and reconstructions at all scales
   - Includes bilinear upsampling for comparison

2. **Detailed comparisons**: `{model}_comparison_sample_{i}.png`
   - Full images at each scale
   - Zoom-in regions showing detail
   - Side-by-side with traditional upsampling

3. **Console output**: Quantitative metrics
   - PSNR, SSIM, MSE, MAE for 32x32 full field reconstruction

## Interpreting Results

### Success Indicators ✅

**Full Field Reconstruction (32x32)**:
- PSNR > 22 dB (good)
- PSNR > 24 dB (excellent)
- SSIM > 0.80 (good)
- SSIM > 0.85 (excellent)

**Scale Invariance (64x64, 96x96)**:
- Sharper than bilinear upsampling of 32x32
- No grid-aligned artifacts
- Smooth, natural appearance
- Consistent colors and structures across scales
- Fine details emerge at higher resolutions

### Failure Indicators ❌

**Poor Full Field Reconstruction**:
- PSNR < 20 dB
- SSIM < 0.75
- Blurry or distorted reconstructions
- Missing colors or structures

**No Scale Invariance**:
- 64x64/96x96 looks worse than bilinear upsampling
- Grid artifacts visible at higher resolutions
- Distorted colors or structures
- Pixelated appearance despite higher resolution

## What You Should See

### Example 1: Good Scale Invariance

```
Ground Truth (32x32)  →  Sparse Input (20%)  →  Reconstructed 32x32
          ↓
   Continuous 64x64    →  Much sharper than 32→64 bilinear
          ↓
   Continuous 96x96    →  Even sharper, more detail emerges
```

**Characteristics**:
- 64x64 and 96x96 look smooth and natural
- Better quality than traditional upsampling
- Details become clearer at higher resolutions
- No artifacts or distortions

### Example 2: Poor Scale Invariance

```
Ground Truth (32x32)  →  Sparse Input (20%)  →  Reconstructed 32x32
          ↓
   Continuous 64x64    →  Similar to or worse than 32→64 bilinear
          ↓
   Continuous 96x96    →  Grid artifacts, distorted
```

**Characteristics**:
- 64x64/96x96 show grid-aligned patterns
- Quality similar to simple upsampling
- Colors or structures distorted
- Model "confused" by off-grid coordinates

## Technical Details

### How It Works

1. **Training**: Model trained on 32x32 images with 20% sparse inputs
   - Input coordinates: 204 pixels (~20% of 1024)
   - Output coordinates: 204 pixels (different 20%)

2. **Full Field Evaluation**: Query all 1024 pixels
   - Tests if model learned full spatial structure
   - Not just interpolating between known points

3. **Multi-Scale Evaluation**: Query at 64x64 (4096 pixels), 96x96 (9216 pixels)
   - Tests continuous representation via Fourier features
   - Should work because coordinates are normalized [0,1]²

### Why Fourier Features Enable Scale Invariance

**Fourier Feature Encoding**:
```python
def fourier_features(coords, num_freqs=256, scale=10.0):
    """
    coords: (N, 2) in [0, 1]² - works for ANY resolution
    """
    B = random_matrix(num_freqs, 2) * scale
    v = 2π * coords @ B.T
    return [sin(v), cos(v)]  # (N, 512)
```

**Key Properties**:
- **Continuous**: Defined for any coordinate in [0,1]²
- **Scale-agnostic**: No dependency on specific resolution
- **High-frequency**: Can represent fine spatial details
- **Smooth**: Small coordinate changes → small feature changes

**Why it works**:
```
32x32 grid:    coords = [0/31, 1/31, ..., 31/31]
64x64 grid:    coords = [0/63, 1/63, ..., 63/63]
96x96 grid:    coords = [0/95, 1/95, ..., 95/95]

All normalized to [0,1]² → Fourier features handle all scales!
```

## Comparison with Traditional Methods

| Method | 32x32 Quality | 64x64 Quality | 96x96 Quality | Continuous |
|--------|---------------|---------------|---------------|------------|
| **Our Model (Continuous Field)** | Native | Native | Native | ✅ Yes |
| **CNN (Fixed Grid)** | Native | Must train new model | Must train new model | ❌ No |
| **Bilinear Upsampling** | Native | Blurry | Very blurry | ❌ No |
| **Bicubic Upsampling** | Native | Smooth but blurry | Blurrier | ❌ No |
| **Super-Resolution (separate model)** | N/A | Good | Requires 3rd model | ❌ No |

**Our Advantage**: Single model, trained once on 32x32, works natively at ALL resolutions!

## Advanced: Theoretical Analysis

### Resolution Scaling

**Memory Complexity**:
- 32x32: 1,024 queries
- 64x64: 4,096 queries (4x)
- 96x96: 9,216 queries (9x)
- 128x128: 16,384 queries (16x)

**Computational Complexity**:
- Local Implicit: O(N_out × N_local) per query
- MAMBA: O(N_out × N_in) via cached einsum

**Practical Limits**:
- Can scale to 256x256 or 512x512 (memory permitting)
- Quality degrades beyond ~4x upsampling without additional training
- Best results: 1-3x upsampling range

### Expected Performance

| Resolution | PSNR (expected) | Quality |
|------------|-----------------|---------|
| 32x32 (native) | 24-26 dB | Excellent |
| 64x64 (2x) | 22-24 dB | Good |
| 96x96 (3x) | 20-22 dB | Fair |
| 128x128 (4x) | 18-20 dB | Decreasing |

**Note**: PSNR can only be computed at 32x32 (where we have ground truth). Higher resolutions are assessed qualitatively.

## Troubleshooting

### Issue: Model checkpoint not found

**Solution**:
```bash
# Save model during training
torch.save(model, 'model_checkpoint.pth')

# Or save state dict
torch.save(model.state_dict(), 'model_state.pth')
```

### Issue: Out of memory at high resolutions

**Solution**:
```bash
# Reduce batch size
python evaluate_multiscale.py --batch_size 4 ...

# Or evaluate fewer scales at once
python evaluate_multiscale.py --scales 32 64 ...
python evaluate_multiscale.py --scales 96 128 ...
```

### Issue: Poor quality at higher resolutions

**Possible causes**:
1. Model didn't converge well (check training loss)
2. Fourier features scale too small (try scale=20 or 30)
3. Local radius too small (Local Implicit only)
4. Not enough training epochs

**Solutions**:
- Train longer (100+ epochs)
- Increase Fourier feature scale
- Use more ODE steps (--num_steps 200)

## References

1. **Fourier Features**: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020

2. **Implicit Neural Representations**: Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020

3. **Continuous Fields**: Park et al., "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation", CVPR 2019

## Citation

If you use this evaluation in your research, please cite:

```bibtex
@misc{nfdiffusion2024,
  title={Scale-Invariant Sparse Field Reconstruction with Implicit Neural Representations},
  author={Your Name},
  year={2024}
}
```
