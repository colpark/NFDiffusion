# Checkerboard Artifact Fix Documentation

## Problem Analysis

Checkerboard artifacts in zero-shot super-resolution are caused by:

### 1. **Nearest Neighbor Upsampling** (Primary Cause)
```python
# PROBLEM: Original v1 upSample
class upSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # ← Creates block patterns!
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1)
        )
```

**Why it fails:**
- Nearest neighbor creates **2×2 blocks of identical pixels**
- Even after 3×3 conv, these block patterns persist
- At higher resolutions (64×64, 96×96), artifacts compound
- Model learns to rely on discrete 32×32 structure, not continuous coordinates

### 2. **Insufficient High-Frequency Encoding**
```python
# v1 settings
COORD_NUM_FREQ = 10  # Only captures up to 2^9 = 512 cycles
COORD_SCALE = 10.0   # May be too aggressive
```

**Why it fails:**
- 10 frequencies may not encode fine details for 3× super-resolution
- 96×96 output needs high-frequency components
- Scale 10.0 can cause aliasing at high resolutions

### 3. **Suboptimal Sparse Input Interpolation**
```python
# In DDIM_Sampler.sample()
sparse_target = F.interpolate(sparse_input, size=(H, W),
                             mode='bilinear', align_corners=False)
```

**Why it fails:**
- Bilinear is piecewise linear, not smooth
- Can introduce slight misalignments at higher resolutions
- Bicubic provides smoother gradients

---

## Solutions Implemented

### **Fix 1: PixelShuffle Upsampling** ⭐ (Critical)

```python
class upSample(nn.Module):
    """Fixed upsampling with PixelShuffle to avoid checkerboard artifacts."""
    def __init__(self, dim_in):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),  # Expand channels
            nn.PixelShuffle(upscale_factor=2),  # Rearrange: 4C,H,W → C,2H,2W
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1)  # Refinement
        )
```

**How PixelShuffle works:**
```
Input:  (B, C, H, W)
Conv:   (B, 4C, H, W)  # 4× channels
Shuffle: (B, C, 2H, 2W)  # Rearrange sub-pixels spatially
Refine: (B, C, 2H, 2W)  # Smooth with conv
```

**Why it fixes checkerboards:**
- Sub-pixel convolution: learns **spatial arrangement** of high-res features
- No duplication: each output pixel comes from **different learned values**
- Smooth gradients: convolution before shuffle ensures continuity
- Used in ESPCN, EDSR, and other state-of-the-art super-resolution

---

### **Fix 2: Increased Coordinate Frequencies**

```python
# v2 settings (improved)
COORD_NUM_FREQ = 16  # Increased from 10
COORD_SCALE = 8.0    # Reduced from 10.0

# Encoding dimension: 4 * 16 = 64 (was 40)
```

**Benefits:**
- **16 frequencies** → captures up to 2^15 = 32,768 cycles
- More high-frequency components for fine details at 96×96
- Scale 8.0 is gentler, reduces aliasing

**Frequency bands:**
```python
# v1: [π*10, 2π*10, 4π*10, ..., 512π*10]
# v2: [π*8,  2π*8,  4π*8,  ..., 32768π*8]
```

---

### **Fix 3: Bicubic Sparse Input Upsampling**

```python
# In DDIM_Sampler.sample()
sparse_target = F.interpolate(sparse_input, size=(H, W),
                             mode='bicubic',  # Changed from bilinear
                             align_corners=False)
```

**Benefits:**
- Bicubic uses **4×4 pixel neighborhoods** (bilinear uses 2×2)
- Smoother interpolation, better preserves gradients
- Less staircase artifacts at high resolutions

---

## Expected Improvements

### Qualitative
- ✅ **No checkerboard patterns** at 64×64 and 96×96
- ✅ **Smoother textures** with continuous gradients
- ✅ **Better edge preservation** without block artifacts
- ✅ **More natural super-resolution** aligned with continuous coordinates

### Quantitative
- **PSNR**: Expect 1-2 dB improvement at 64×64, 2-3 dB at 96×96
- **SSIM**: Structural similarity should increase
- **Visual quality**: Significant reduction in artifacts

---

## Comparison: v1 vs v2

| Component | v1 (Original) | v2 (Fixed) |
|-----------|---------------|------------|
| **Upsampling** | Nearest neighbor | **PixelShuffle** |
| **Coord frequencies** | 10 | **16** |
| **Coord scale** | 10.0 | **8.0** |
| **Coord encoding dim** | 40 | **64** |
| **Sparse interpolation** | Bilinear | **Bicubic** |
| **Checkerboards?** | ❌ Yes | ✅ **No** |

---

## Training Recommendations

### Hyperparameters
```python
# Same as v1, but may converge faster
STEPS = 50_000  # Can reduce if quality good earlier
BATCH_TRAIN = 64
LR = 2e-4
```

### Validation
- Check 64×64 and 96×96 outputs at VIS_EVERY intervals
- Look for:
  - Smooth gradients (no block patterns)
  - Continuous textures
  - Natural edges (no staircase artifacts)

### Early Stopping
- If 32×32 reconstruction is good but super-resolution has artifacts:
  - Increase COORD_NUM_FREQ to 20
  - Further reduce COORD_SCALE to 6.0
  - Add dropout=0.1 to prevent overfitting to 32×32 only

---

## Alternative Fixes (If Issues Persist)

### Option A: Transpose Convolution
```python
class upSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_in, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1)
        )
```

### Option B: Bilinear + Conv (Simpler)
```python
class upSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1)  # Extra refinement
        )
```

### Option C: Multi-Scale Training
Add resolution diversity during training:
```python
# In training loop
if random.random() < 0.2:
    # Randomly crop to 24×24 or 28×28
    crop_size = random.choice([24, 28])
    x, m_cond, m_supv = random_crop(x, m_cond, m_supv, crop_size)
    coords = make_coordinate_grid(B, crop_size, crop_size, DEVICE)
else:
    coords = make_coordinate_grid(B, 32, 32, DEVICE)
```

---

## Debugging Checklist

If checkerboards still appear:

1. **Verify upSample implementation:**
   ```python
   # Test upsampling
   x = torch.randn(1, 64, 16, 16).to(DEVICE)
   up = upSample(64).to(DEVICE)
   y = up(x)
   print(y.shape)  # Should be (1, 64, 32, 32)

   # Visualize pattern
   import matplotlib.pyplot as plt
   plt.imshow(y[0,0].cpu().numpy(), cmap='gray')
   plt.title("Should be smooth, not blocky")
   ```

2. **Check coordinate encoding:**
   ```python
   coords = make_coordinate_grid(1, 96, 96, DEVICE)
   features = coord_encoder(coords)
   print(features.shape)  # Should be (1, 96, 96, 64)

   # Visualize first frequency
   plt.imshow(features[0, :, :, 0].cpu().numpy())
   plt.title("Should be smooth sine wave")
   ```

3. **Inspect DDIM outputs step-by-step:**
   ```python
   # In DDIM sampling loop, save intermediate steps
   if i % 10 == 0:
       save_image(xt, f"step_{i}.png")
   # Check when artifacts first appear
   ```

4. **Compare with bicubic baseline:**
   ```python
   # Simple bicubic upsampling of 32×32 reconstruction
   recon_32_upsampled = F.interpolate(recon_32, size=(96, 96), mode='bicubic')
   # Should be smoother than model output if model has issues
   ```

---

## References

- **PixelShuffle**: Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" (CVPR 2016)
- **Checkerboard artifacts**: Odena et al., "Deconvolution and Checkerboard Artifacts" (Distill 2016)
- **Fourier features**: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (NeurIPS 2020)
