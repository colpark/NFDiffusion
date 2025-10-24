# MAMBA v1 Problems Analysis

**Observation**: Generated images at all scales (32x32, 64x64, 96x96) are noisy/wiggly, though noise follows color patterns.

**Goal**: Identify root causes sorted by likelihood before implementing v2 improvements.

---

## HIGH PROBABILITY Issues (>70% likelihood)

### 1. **Insufficient ODE Solver Steps** ⭐⭐⭐⭐⭐

**Likelihood**: 90%

**Problem**:
- Current: `num_steps=50` during training, evaluation
- Flow matching requires integrating the learned velocity field from t=0 to t=1
- Too few steps → poor numerical integration → residual noise

**Evidence**:
- Standard diffusion models use 100-1000 steps for high quality
- Heun's method is 2nd order but still needs sufficient steps
- dt = 1/50 = 0.02 is quite large for ODE integration

**Expected Impact**:
- Increasing to 100-200 steps should significantly reduce noise
- Diminishing returns beyond ~200 steps

**Test**:
```python
# Compare sampling with different step counts
for num_steps in [50, 100, 150, 200]:
    pred = heun_sample(model, coords, input_coords, input_values, num_steps=num_steps)
    # Visual quality should improve dramatically
```

---

### 2. **Training Not Fully Converged** ⭐⭐⭐⭐

**Likelihood**: 80%

**Problem**:
- Current: 100 epochs with cosine annealing
- Learning rate decays to near-zero by epoch 100
- Model might not have learned optimal velocity field

**Evidence**:
- Complex continuous field reconstruction is difficult
- MAMBA has 18M parameters - needs time to converge
- No early stopping based on validation plateau

**Expected Impact**:
- Training for 200-300 epochs could improve smoothness
- May need to adjust scheduler (restart, warmup, or slower decay)

**Test**:
```python
# Check if loss is still decreasing
plt.plot(losses)
# If loss hasn't plateaued, train longer
```

---

### 3. **Fourier Feature Configuration** ⭐⭐⭐⭐

**Likelihood**: 75%

**Problem**:
- Current: `num_fourier_feats=256, scale=10.0`
- Scale controls frequency spectrum of encoded coordinates
- Wrong scale → either too high-frequency (noise) or too low-frequency (blur)

**Evidence**:
- Tancik et al. showed scale critically affects quality
- scale=10 might be encoding too high frequencies for 32x32 images
- Could be causing model to fit noise instead of structure

**Diagnosis**:
```python
# Visualize Fourier features
coords = torch.linspace(0, 1, 100)
for scale in [1.0, 5.0, 10.0, 20.0]:
    feats = fourier_features(coords, scale=scale)
    plt.plot(feats[0])  # Check frequency content
```

**Expected Impact**:
- Lowering scale to 3-5 might reduce high-freq noise
- Or using learnable scale (optimize during training)

---

### 4. **SSM State Dimension Too Small** ⭐⭐⭐⭐

**Likelihood**: 70%

**Problem**:
- Current: `d_state=16` (SSM hidden state dimension)
- State vector h ∈ ℝ^16 might be insufficient bottleneck
- Long-range spatial dependencies compressed into 16 dims

**Evidence**:
- Each pixel needs to attend to ~204 sparse inputs
- 16-dim state might lose information → noise artifacts
- Original Mamba paper uses 16-256 depending on task

**Expected Impact**:
- Increasing to d_state=32 or 64 could improve quality
- Trade-off: more memory and slower training

**Test**:
```python
# Try different state sizes
for d_state in [16, 32, 64]:
    model = MAMBADiffusion(d_state=d_state)
    # Train and compare quality vs speed
```

---

## MEDIUM PROBABILITY Issues (40-70% likelihood)

### 5. **Flow Matching Formulation** ⭐⭐⭐

**Likelihood**: 60%

**Problem**:
- Current: Straight-line flow `x_t = (1-t)x_0 + t*x_1`
- Velocity target: `v_t = x_1 - x_0` (constant)
- This is simplest but not necessarily optimal

**Alternative Formulations**:
1. **Optimal Transport Flow**: Minimize transport cost
2. **Conditional Flow**: Learn time-dependent paths
3. **Rectified Flow**: Iteratively straighten trajectories

**Evidence**:
- Straight-line paths can be suboptimal in high dimensions
- Optimal transport could reduce noise in reconstruction

**Expected Impact**:
- Optimal transport flow might give smoother results
- Requires more complex training (Sinkhorn iterations)

---

### 6. **Learning Rate Schedule** ⭐⭐⭐

**Likelihood**: 55%

**Problem**:
- Current: `lr=1e-4` with `CosineAnnealingLR`
- LR decays to ~0 by epoch 100
- Might converge to suboptimal local minimum

**Issues**:
```python
# At epoch 100:
lr_final = 1e-4 * cos(π * 100/100) ≈ 1e-4 * (-1) ≈ 0
# Model effectively stops learning around epoch 80-90
```

**Better Options**:
1. **Cosine with warmup and restarts**
2. **Constant LR with step decay**
3. **ReduceLROnPlateau** (adaptive)

**Expected Impact**:
- Better schedule could improve final quality by 10-20%

---

### 7. **MAMBA Architecture Mismatch** ⭐⭐⭐

**Likelihood**: 50%

**Problem**:
- MAMBA designed for **sequential** data (text, audio)
- Spatial 2D coordinates don't have natural sequence order
- Flattening spatial grid might lose spatial structure

**Evidence**:
- Local Implicit uses spatial attention (distance-aware)
- MAMBA treats pixels as sequence → might miss 2D relationships
- State propagation is 1D (time) not 2D (space)

**Fundamental Issue**:
```python
# Coordinates are unordered spatial positions
input_coords: (B, 204, 2)  # 2D positions
# MAMBA processes as sequence
SSM: h[t] = A*h[t-1] + B*x[t]  # Sequential, not spatial
```

**Expected Impact**:
- Adding 2D spatial inductive bias could help
- Or hierarchical/multi-scale MAMBA

---

### 8. **No Explicit Smoothness Regularization** ⭐⭐⭐

**Likelihood**: 50%

**Problem**:
- Current loss: `MSE(v_pred, v_target)` only
- No penalty for high-frequency noise
- Model can overfit to noisy patterns

**Missing Regularization**:
1. **Gradient penalty**: Penalize ∇v being too large
2. **Laplacian smoothness**: Encourage smooth solutions
3. **Total variation**: Reduce sharp changes

**Implementation**:
```python
# Add smoothness loss
def smoothness_loss(pred_images):
    # Gradient in x and y
    dx = pred_images[:, :, 1:, :] - pred_images[:, :, :-1, :]
    dy = pred_images[:, :, :, 1:] - pred_images[:, :, :, :-1]
    return (dx.abs().mean() + dy.abs().mean())

total_loss = mse_loss + 0.01 * smoothness_loss(pred)
```

**Expected Impact**:
- Should reduce high-frequency noise artifacts
- May slightly reduce fine detail

---

### 9. **State Clamping Side Effects** ⭐⭐⭐

**Likelihood**: 45%

**Problem**:
- Current: `h = torch.clamp(h, min=-10.0, max=10.0)`
- Prevents explosion but creates hard boundaries
- Might cause discontinuities in learned function

**Evidence**:
```python
# If state hits boundaries frequently
h_clamped = clamp(h)  # Many values at -10 or +10
# Gradient flow is cut off → learning issues
```

**Better Alternatives**:
1. **Softer clamping**: `h = tanh(h/10) * 10`
2. **Gradient clipping instead of value clipping**
3. **Better initialization** (reduce need for clamping)

**Expected Impact**:
- Removing or softening clamp might reduce artifacts
- Need to ensure training remains stable

---

## LOWER PROBABILITY Issues (20-40% likelihood)

### 10. **Batch Size Effects** ⭐⭐

**Likelihood**: 35%

**Problem**:
- Current: `batch_size=64`
- Larger batches → more stable gradients
- Smaller batches → more noise in updates

**Trade-offs**:
- Larger batches: Better gradient estimates, but slower per-epoch
- Smaller batches: More updates, but noisier

**Expected Impact**: Minimal (64 is reasonable)

---

### 11. **Time Embedding Quality** ⭐⭐

**Likelihood**: 30%

**Problem**:
- Current: Sinusoidal time embedding
- Might not be expressive enough for flow matching

**Alternative**:
```python
# Learned time embedding with MLP
self.time_embed = nn.Sequential(
    SinusoidalEmbedding(d_model),
    nn.Linear(d_model, d_model * 4),
    nn.SiLU(),
    nn.Linear(d_model * 4, d_model)
)
```

**Expected Impact**: Small improvement (5-10%)

---

### 12. **Heun vs Higher-Order Solvers** ⭐⭐

**Likelihood**: 25%

**Problem**:
- Heun is 2nd order (good), but not the best
- RK4 is 4th order (better but slower)
- Adaptive solvers (DoPri5) auto-adjust step size

**Comparison**:
```python
# Heun: O(h²) error, 2 function evals per step
# RK4:  O(h⁴) error, 4 function evals per step
# For same quality: Heun needs 2-3x more steps than RK4
```

**Expected Impact**:
- RK4 with 50 steps ≈ Heun with 100-150 steps
- Probably not the main issue

---

### 13. **Data Augmentation** ⭐

**Likelihood**: 20%

**Problem**:
- No augmentation during training
- Model sees same images every epoch
- Might overfit to specific pixel patterns

**Potential Augmentations**:
- Random flips (horizontal/vertical)
- Small rotations
- Color jittering

**Expected Impact**: Modest generalization improvement

---

## UNLIKELY Issues (<20% likelihood)

### 14. **Numerical Precision** ⭐

**Likelihood**: 15%

**Problem**: FP32 should be sufficient for this task

---

### 15. **Random Seed Effects** ⭐

**Likelihood**: 10%

**Problem**: Would affect reproducibility, not systematic noise

---

## Recommended Testing Priority

### Phase 1: Quick Wins (Test First)
1. **Increase ODE steps**: 50 → 100 → 200 (immediate test)
2. **Train longer**: 100 → 200 epochs (overnight run)
3. **Adjust Fourier scale**: Try 3.0, 5.0, 10.0, 20.0

### Phase 2: Architectural Changes
4. **Increase d_state**: 16 → 32 → 64
5. **Add smoothness regularization**
6. **Improve learning rate schedule**

### Phase 3: Advanced Improvements
7. **Optimal transport flow matching**
8. **Spatial inductive bias for MAMBA**
9. **Alternative solvers (RK4, adaptive)**

---

## Diagnostic Experiments

### Experiment 1: ODE Step Sweep
```python
num_steps_list = [50, 100, 150, 200, 250]
for ns in num_steps_list:
    pred = heun_sample(model, coords, inputs, num_steps=ns)
    psnr = compute_psnr(pred, gt)
    print(f"Steps: {ns}, PSNR: {psnr:.2f} dB")
```

**Expected**: PSNR should increase significantly with steps

---

### Experiment 2: Fourier Scale Sweep
```python
for scale in [1.0, 3.0, 5.0, 10.0, 20.0]:
    model = MAMBADiffusion(fourier_scale=scale)
    train(model)
    quality = evaluate(model)
```

**Expected**: Sweet spot around 3-5 for 32x32 images

---

### Experiment 3: Training Convergence Check
```python
# Train for 300 epochs with plateau detection
train_losses, val_losses = [], []
for epoch in range(300):
    train_loss = train_epoch()
    val_loss = validate()
    if val_loss hasn't improved for 50 epochs:
        break
```

**Expected**: If loss plateaus before 100 epochs → overfitting
If still decreasing at 100 → undertrained

---

### Experiment 4: State Size Ablation
```python
for d_state in [8, 16, 32, 64, 128]:
    model = MAMBADiffusion(d_state=d_state)
    quality, speed = train_and_evaluate(model)
    print(f"d_state={d_state}: PSNR={quality:.2f}, Time={speed:.2f}s")
```

**Expected**: Quality improves with d_state up to a point

---

## Summary

**Most Likely Culprits** (address first):
1. ⭐⭐⭐⭐⭐ Too few ODE solver steps (50 → 200)
2. ⭐⭐⭐⭐ Training not converged (100 → 200-300 epochs)
3. ⭐⭐⭐⭐ Fourier scale misconfigured (try 3-5)
4. ⭐⭐⭐⭐ SSM state too small (16 → 32-64)

**Quick Tests** (run immediately):
- Increase num_steps during sampling only (no retraining needed)
- Train existing model for another 100 epochs
- Check loss curves for convergence

**Next Steps**:
Once we identify the main issues through diagnostics, we can implement targeted fixes in v2_mamba_improve.
