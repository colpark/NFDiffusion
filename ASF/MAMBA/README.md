# MAMBA Flow Matching for Sparse Field Reconstruction

**Version**: v1 - Flow Matching Implementation
**Date**: October 2024
**Architecture**: State Space Models (SSM) with Selective Gating

## Overview

This folder contains a complete implementation of MAMBA (Selective State Space Models) applied to sparse field reconstruction using flow matching diffusion. The model learns continuous implicit neural representations from sparse CIFAR-10 data.

## Files

### Main Implementation
- **`mamba_diffusion.ipynb`** - Complete training notebook
  - MAMBA SSM architecture with selective gating
  - Flow matching diffusion training
  - Multi-scale evaluation (32x32, 64x64, 96x96)
  - Model saving (best & latest checkpoints)
  - Full field visualization during training

### Documentation
- **`MAMBA_FIXES.md`** - Numerical stability fixes
  - Zero-order hold discretization
  - Safe division handling
  - State clamping to prevent explosion
  - Parameter initialization strategies

- **`PERFORMANCE_ANALYSIS.md`** - Performance optimization
  - Python loop bottleneck analysis
  - Vectorized einsum implementation (5x speedup)
  - O(N²) memory trade-off for parallelization
  - Benchmark comparisons

- **`mamba_architecture_report.html`** - Interactive architecture guide
  - SSM mathematical foundations
  - Selective gating mechanism
  - Continuous → discrete transformation
  - Comparison with Local Implicit Fields

## Architecture

### State Space Model (SSM)

**Continuous Form**:
```
h'(t) = A h(t) + B x(t)
y(t) = C h(t) + D x(t)
```

**Discrete Form (Zero-Order Hold)**:
```
h_k = exp(Δt·A) h_{k-1} + [(exp(Δt·A)-I)/A] B x_k
y_k = C h_k + D x_k
```

### Key Features

1. **Selective Gating**: Input-dependent B and C matrices
2. **Linear Complexity**: O(N) in theory, O(N²) in practice for parallelization
3. **Long-Range Dependencies**: State propagation across sequence
4. **Fourier Features**: Continuous coordinate encoding
5. **Flow Matching**: Straight-line diffusion training

## Model Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Model dimension |
| d_state | 16 | SSM state dimension |
| num_layers | 6 | Number of MAMBA blocks |
| expand_factor | 2 | Channel expansion ratio |
| num_fourier_feats | 256 | Fourier feature dimension |
| Parameters | ~18M | Total trainable parameters |

## Training

### Dataset
- **CIFAR-10** (32×32 RGB images)
- **Input**: 20% sparse pixels (~204 pixels)
- **Output**: Different 20% sparse pixels (~204 pixels)
- **Goal**: Reconstruct full 1024-pixel field

### Training Configuration
```python
epochs = 100
learning_rate = 1e-4
batch_size = 64
optimizer = Adam
scheduler = CosineAnnealingLR
num_steps = 50  # ODE solver steps
```

### Model Saving
- **Best model**: `checkpoints/mamba_best.pth` (lowest validation loss)
- **Latest model**: `checkpoints/mamba_latest.pth` (most recent epoch)

## Performance

### Numerical Stability Fixes
- **Issue**: NaN values during training
- **Solution**: Proper discretization, state clamping, careful initialization
- **Result**: Stable training to convergence

### Speed Optimization
- **Original**: ~400ms per forward pass (Python for loop)
- **Optimized**: ~80ms per forward pass (vectorized einsum)
- **Speedup**: 5x faster
- **Trade-off**: O(N²) memory (~640 MB) for parallelization

### Comparison
| Model | Speed | Complexity | Memory |
|-------|-------|------------|--------|
| Local Implicit | ~50ms | O(N²) | O(N²) |
| MAMBA (old) | ~400ms | O(N) | O(N) |
| **MAMBA (fast)** | **~80ms** | **O(N²)** | **O(N²)** |

## Multi-Scale Evaluation

Tests scale invariance via Fourier features:

### Resolutions Tested
- **32×32** (native training resolution) - 1,024 pixels
- **64×64** (2× upsampling) - 4,096 pixels
- **96×96** (3× upsampling) - 9,216 pixels

### Expected Results
**Success Indicators**:
- PSNR > 24 dB, SSIM > 0.85 at 32×32
- Sharper than bilinear upsampling at higher resolutions
- No grid artifacts
- Smooth, natural appearance

**Failure Indicators**:
- Grid artifacts at 64×64/96×96
- Quality similar to or worse than traditional upsampling
- Distorted colors or structures

## Usage

### Training
```python
# Load notebook and run cells
# Training will automatically:
# - Save best and latest models
# - Generate visualizations every 5 epochs
# - Evaluate every 2 epochs
# - Show full field reconstruction during training
```

### Loading Trained Model
```python
checkpoint = torch.load('checkpoints/mamba_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded from epoch {checkpoint['epoch']}")
print(f"Best val loss: {checkpoint['best_val_loss']:.6f}")
```

### Multi-Scale Evaluation
```bash
# Standalone script
python ../evaluate_multiscale.py \
    --model mamba \
    --checkpoint checkpoints/mamba_best.pth \
    --scales 32 64 96 128 \
    --num_steps 100 \
    --output_dir ./results_mamba

# Or run evaluation cells in notebook (Section 6)
```

## Key Innovations

1. **Numerical Stability**: Fixed SSM discretization issues
2. **Performance**: 5× speedup via vectorization
3. **Scale Invariance**: Tests continuous representation learning
4. **Model Saving**: Automatic best/latest checkpoint management
5. **Full Field Viz**: Monitor complete reconstruction during training

## Technical Challenges Solved

### Challenge 1: NaN During Training
- **Cause**: Improper discretization, unbounded state accumulation
- **Solution**: Zero-order hold, state clamping, bounded A matrix
- **Documentation**: `MAMBA_FIXES.md`

### Challenge 2: Slow Training Speed
- **Cause**: Python for loop (408 sequential iterations)
- **Solution**: Vectorized einsum with pre-computed decay matrix
- **Documentation**: `PERFORMANCE_ANALYSIS.md`

### Challenge 3: Memory vs Speed
- **Decision**: O(N²) memory trade-off acceptable for N=408
- **Rationale**: ~640 MB for 5× speedup is reasonable
- **Alternative**: Chunk-based processing for larger sequences

## Hypothesis Testing

**Hypothesis**: If MAMBA learned truly continuous representations via Fourier features, it should:
1. Reconstruct full 32×32 field from 20% sparse inputs
2. Generalize to 64×64, 96×96 without retraining
3. Produce sharper results than traditional upsampling
4. Show no grid-aligned artifacts at higher resolutions

**Validation**: Multi-scale evaluation in Section 6 of notebook

## Future Improvements

1. **Custom CUDA Kernels**: True O(N) performance for N > 1000
2. **FlashAttention-style Kernel**: Memory-efficient SSM computation
3. **Hierarchical SSM**: Multi-scale state propagation
4. **Mixed Precision**: FP16 for 2× memory reduction
5. **Gradient Checkpointing**: Trade compute for memory

## References

1. **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
2. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
3. **Fourier Features**: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions", NeurIPS 2020
4. **State Space Models**: Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces", ICLR 2022

## Version History

- **v1** (Oct 2024): Initial flow matching implementation with numerical fixes and performance optimization

## Citation

```bibtex
@misc{mamba_flow_matching_v1,
  title={MAMBA Flow Matching for Sparse Field Reconstruction},
  author={NFDiffusion Team},
  year={2024},
  note={v1 - Numerically stable, performance optimized}
}
```

## Contact & Support

For issues, questions, or contributions, see main repository documentation.
