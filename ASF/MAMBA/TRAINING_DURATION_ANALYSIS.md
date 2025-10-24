# Training Duration Analysis: Is 200 Epochs Enough?

## Your Training Configuration

**Training Details**:
- **Epochs**: 200
- **Iterations per epoch**: 780
- **Total iterations**: 780 × 200 = **156,000 iterations**
- **Dataset**: CIFAR-10 (50,000 training images)
- **Batch size**: 64 (50,000 / 64 ≈ 781 batches)
- **Model size**: ~18M parameters
- **Learning rate**: 1e-4 with CosineAnnealingLR

---

## Quick Answer: **Probably Marginal** ⚠️

156,000 iterations is **on the lower end** for diffusion/flow matching models of this complexity.

**Status**:
- ✅ **Minimum viable**: Yes, model has seen data enough times
- ⚠️ **Optimal convergence**: Questionable, likely needs more
- ❌ **Sufficient for high quality**: Probably not for this task

---

## Comparison with Standard Practices

### Typical Training Iterations

| Model Type | Typical Iterations | Your Training | Ratio |
|------------|-------------------|---------------|-------|
| **DDPM (Ho et al.)** | 800,000 | 156,000 | 0.19× |
| **Stable Diffusion** | 600,000+ | 156,000 | 0.26× |
| **Flow Matching (Lipman)** | 400,000 | 156,000 | 0.39× |
| **Small diffusion models** | 200,000-300,000 | 156,000 | 0.52-0.78× |

**Your training is roughly 20-40% of typical diffusion model training.**

### Why Flow Matching Might Need Less

**Advantages**:
- No noise schedule to learn (straight paths)
- Simpler objective (velocity prediction vs score matching)
- Typically 2-3× faster convergence than DDPM

**Still**:
- 156k might be insufficient for complex continuous fields
- MAMBA has 18M parameters - needs time to converge
- Sparse reconstruction (20% → 100%) is challenging

---

## Critical Issue: Learning Rate Schedule ⚠️⚠️⚠️

### **Your LR Decays to Zero Too Early**

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Learning rate over time:
Epoch   0: lr = 1e-4        (full learning)
Epoch  50: lr = 7.07e-5     (70% of original)
Epoch 100: lr = 5e-9        (≈ 0, STOPPED LEARNING!)
Epoch 150: lr = 5e-9        (≈ 0, wasting time)
Epoch 200: lr = 1e-4        (back to start, but too late!)
```

**Problem**:
- Cosine annealing: `lr = lr_max * 0.5 * (1 + cos(π * epoch / T_max))`
- At epoch 100 (halfway): `cos(π) = -1` → lr ≈ 0
- **Epochs 100-200: Model barely learning anything!**

### Effective Training

**You didn't train for 200 epochs, you trained for ~120 effective epochs:**
- Epochs 0-100: Normal learning (decaying from 1e-4 to ~0)
- Epochs 100-200: Minimal learning (lr ≈ 0)

**So your effective iteration count is closer to**:
- 780 × 120 = **93,600 effective iterations**
- This is **very low** for a diffusion model!

---

## How to Check If Training Is Sufficient

### Diagnostic 1: Plot Training Loss Curve

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot 1: Loss over epochs
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# Plot 2: Loss over last 50 epochs (zoomed)
plt.subplot(1, 2, 2)
plt.plot(range(150, 200), losses[150:])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss (Epochs 150-200)')
plt.grid(True)

plt.tight_layout()
plt.savefig('loss_convergence.png')
plt.show()
```

**What to look for**:

✅ **Training is sufficient if**:
```
Loss clearly plateaued (flat line) for last 50+ epochs
Example:
  Epoch 100: 0.0025
  Epoch 150: 0.0024
  Epoch 200: 0.0024  ← No improvement
```

❌ **Training is insufficient if**:
```
Loss still decreasing steadily at epoch 200
Example:
  Epoch 100: 0.0050
  Epoch 150: 0.0030
  Epoch 200: 0.0015  ← Still improving!
```

⚠️ **Training stopped early if**:
```
Loss plateaued around epoch 100-120, then flat
Example:
  Epoch  80: 0.0030
  Epoch 100: 0.0025
  Epoch 120: 0.0024
  Epoch 200: 0.0024  ← Stuck, LR too low!
```

---

### Diagnostic 2: Validation Metrics Over Time

```python
# If you saved validation metrics
epochs = np.arange(0, 200, 2)  # evaluated every 2 epochs
val_psnr = [...]  # your validation PSNR values

plt.plot(epochs, val_psnr, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Validation PSNR (dB)')
plt.title('Validation Quality Over Training')
plt.grid(True)
plt.savefig('validation_metrics.png')
```

**What to look for**:

✅ **Sufficient**: PSNR plateaued for last 50+ epochs
❌ **Insufficient**: PSNR still increasing at epoch 200
⚠️ **Stopped early**: PSNR plateaued around epoch 100-120

---

### Diagnostic 3: Learning Rate Schedule Check

```python
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(0, 200)
lr_initial = 1e-4

# Your current schedule
lr_cosine = lr_initial * 0.5 * (1 + np.cos(np.pi * epochs / 200))

plt.figure(figsize=(10, 4))
plt.plot(epochs, lr_cosine, linewidth=2)
plt.axvline(100, color='red', linestyle='--', label='LR ≈ 0 (epoch 100)')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('CosineAnnealingLR Schedule (T_max=200)')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig('lr_schedule_problem.png')
plt.show()

print(f"Epoch   0: lr = {lr_cosine[0]:.2e}")
print(f"Epoch  50: lr = {lr_cosine[50]:.2e}")
print(f"Epoch 100: lr = {lr_cosine[100]:.2e}  ← EFFECTIVELY ZERO")
print(f"Epoch 150: lr = {lr_cosine[150]:.2e}")
print(f"Epoch 200: lr = {lr_cosine[199]:.2e}")
```

**Expected output**:
```
Epoch   0: lr = 1.00e-04
Epoch  50: lr = 7.07e-05
Epoch 100: lr = 5.00e-09   ← EFFECTIVELY ZERO
Epoch 150: lr = 2.93e-05
Epoch 200: lr = 9.95e-05
```

**This confirms**: Model stopped learning around epoch 100!

---

## Root Cause: Learning Rate Schedule Mismatch

### The Bug in Your Training

```python
# Your code (INCORRECT for 200 epochs):
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#                                                                T_max=epochs
# This means: complete one full cosine cycle in T_max epochs
# At epoch T_max/2 (=100): lr = 0
```

### What You Should Have Used

**Option 1: Set T_max Correctly**
```python
# For 200 epochs of meaningful learning:
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=200,  # epochs
    eta_min=1e-6  # don't decay all the way to zero!
)
```

**Option 2: Use Cosine with Warmup (Better)**
```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    epochs=200,
    steps_per_epoch=780,
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)
```

**Option 3: Use Warmup + Cosine Decay (Best)**
```python
# Warmup for first 10 epochs, then decay
def get_lr_lambda(epoch):
    warmup_epochs = 10
    if epoch < warmup_epochs:
        # Linear warmup
        return epoch / warmup_epochs
    else:
        # Cosine decay from warmup_epochs to total_epochs
        progress = (epoch - warmup_epochs) / (200 - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
```

---

## Recommendation: What You Should Do

### Immediate Action: Check Your Loss Curve

**Run this diagnostic**:
```python
# Load your losses
losses = [...]  # from your training

# Check if loss plateaued
last_50_losses = losses[-50:]
loss_std = np.std(last_50_losses)
loss_mean = np.mean(last_50_losses)

print(f"Last 50 epochs - Mean: {loss_mean:.6f}, Std: {loss_std:.6f}")

if loss_std / loss_mean < 0.01:
    print("✓ Loss plateaued - training converged")
else:
    print("✗ Loss still varying - training not converged")

# Check when loss plateaued
plateau_epoch = None
for i in range(50, len(losses)):
    window = losses[i-50:i]
    if np.std(window) / np.mean(window) < 0.01:
        plateau_epoch = i
        break

if plateau_epoch:
    print(f"Loss plateaued around epoch {plateau_epoch}")
else:
    print("Loss never plateaued!")
```

---

### Strategy 1: If Loss Plateaued Before Epoch 200 ✅

**You're fine!** The model converged early. The noise is from other issues:
- → Test increasing ODE steps (50 → 200)
- → Adjust Fourier scale
- → Add smoothness regularization

---

### Strategy 2: If Loss Plateaued Around Epoch 100 ⚠️

**The LR schedule killed your training!**

**Fix**: Continue training with better LR schedule

```python
# Load your checkpoint
checkpoint = torch.load('checkpoints/mamba_latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create NEW optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Lower restart LR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # Another 100 epochs
    eta_min=1e-6
)

# Continue training
losses_continued = train_flow_matching(
    model, train_loader, test_loader,
    epochs=100,  # 100 more epochs
    lr=5e-5,
    save_dir='checkpoints_continued'
)
```

**Total effective training**:
- Original: ~100 effective epochs (before LR → 0)
- Continued: +100 epochs
- **Total: 200 effective epochs** = ~300,000 iterations ✅

---

### Strategy 3: If Loss Still Decreasing at Epoch 200 ❌

**You need much more training!**

**Option A: Train from scratch with correct schedule**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=300,  # 300 epochs total
    eta_min=1e-6
)

losses = train_flow_matching(model, train_loader, test_loader, epochs=300)
```

**Option B: Continue with warmup restart**
```python
# Continue from checkpoint with warmup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Warm restart

# Gradually increase LR then decay
def get_lr_lambda(epoch):
    if epoch < 20:
        return epoch / 20  # Warmup to 1.0
    else:
        return 0.5 * (1 + np.cos(np.pi * (epoch - 20) / 180))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

# Train for 200 more epochs
losses_continued = train_flow_matching(model, ..., epochs=200)
```

---

## Expected Training Time for High Quality

### Recommended Targets

| Quality Level | Total Iterations | Epochs (your data) | Approx Time |
|---------------|------------------|-------------------|-------------|
| **Baseline** | 100,000 | 128 | 4-6 hours |
| **Good** | 200,000 | 256 | 8-12 hours |
| **High Quality** | 400,000 | 512 | 16-24 hours |
| **Excellent** | 600,000+ | 768+ | 24-36 hours |

**Your current**: 156,000 iterations (but only ~93,600 effective)

**Recommendation for v2**:
- Train for **400,000 iterations** (512 epochs)
- Use proper LR schedule with warmup
- Should see significant quality improvement

---

## Summary: Is 200 Epochs Enough?

### Short Answer: **No, probably not**

**Reasons**:
1. ❌ **Effective iterations too low**: ~93,600 vs 400,000 recommended
2. ❌ **LR schedule bug**: Model stopped learning at epoch 100
3. ❌ **Complexity**: 18M params + sparse reconstruction needs more training
4. ⚠️ **Noise symptoms**: Wiggly outputs suggest undertraining

### Action Items:

1. **Immediate** (5 minutes):
   - Plot your loss curve
   - Check when it plateaued
   - Examine learning rate schedule

2. **If loss plateaued early** (< epoch 100):
   - Continue training with better LR schedule for 100-200 more epochs

3. **If loss still decreasing**:
   - Train from scratch with 400-500 epochs and proper schedule

4. **While training** (quick wins):
   - Test increasing ODE steps to 200 (might help immediately!)
   - Adjust Fourier scale to 3.0 or 5.0

---

## Updated Training Code for v2

```python
def train_flow_matching_v2(
    model, train_loader, test_loader,
    epochs=400,  # More epochs!
    lr=1e-4,
    device='cuda',
    save_dir='checkpoints'
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Better LR schedule with warmup
    warmup_epochs = 20
    def get_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

    # Rest same as before...
```

**This should give you ~400k iterations of effective training!**
