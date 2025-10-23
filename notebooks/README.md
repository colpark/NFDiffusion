# NFDiffusion Notebooks

Exploratory notebooks for prototyping and analysis of neural field + diffusion architectures.

## Implemented Notebooks

### 01_approach1_latent_diffusion_nf.ipynb
**Status**: ✅ Complete
**Description**: Baseline implementation of Approach 1 (Latent Diffusion + Neural Field Decoder)

**Components**:
- SIREN-based neural field decoder for continuous 1D signal representation
- MLP sparse observation encoder
- End-to-end training on sparse-to-dense reconstruction task

**Key Features**:
- Toy problem: 1D signals (sine waves, Fourier series)
- Sparsity: 20% random observations → full signal reconstruction
- Continuous representation: Query at arbitrary resolution
- Visual validation of learned representations

**Results**:
- Successfully reconstructs 1D signals from sparse observations
- Neural field provides smooth continuous representations
- Foundation for adding diffusion component

---

### 02_approach1_with_diffusion.ipynb
**Status**: ✅ Complete
**Description**: Complete Approach 1 with DDPM diffusion on latent codes

**Components**:
- Sparse encoder (from notebook 01)
- **DDPM diffusion model** on 64D latent space
- **Denoising network** with sinusoidal time embeddings
- SIREN neural field decoder (from notebook 01)

**Architecture**:
```
Training:
Sparse Input → Encoder → z_0 → Add Noise → z_t
                                    ↓
                            Denoiser(z_t, t) → ε_pred
                                    ↓
                            Neural Field → Reconstruction

Generation:
z_T ~ N(0, I) → Reverse Diffusion → z_0 → Neural Field → Output (any resolution)
```

**Key Features**:
- Dual loss training: Reconstruction + Diffusion (noise prediction)
- Cosine noise schedule (100 timesteps)
- Unconditional generation from pure noise
- Arbitrary resolution querying (64 to 1024+ points)

**Results**:
- Successful latent diffusion training
- Generates plausible new 1D signals
- Maintains continuous representation benefits
- Fast sampling (100 steps in latent space)

---

## Quick Start

### Prerequisites
```bash
pip install torch numpy matplotlib tqdm
```

### Running Notebooks
```bash
jupyter notebook
# Open 01_approach1_latent_diffusion_nf.ipynb
# Follow cells sequentially
```

---

## Implementation Roadmap

### ✅ Phase 1: Baseline (Completed)
- [x] 1D toy problem dataset
- [x] SIREN neural field decoder
- [x] Sparse observation encoder
- [x] Reconstruction-only training
- [x] DDPM diffusion on latents
- [x] Denoising network
- [x] End-to-end latent diffusion training
- [x] Arbitrary resolution validation

### 📋 Phase 2: Enhancements (Next)
- [ ] Conditional generation (condition on sparse observations)
- [ ] Transformer-based sparse encoder (better than MLP)
- [ ] DDIM sampling (faster inference)
- [ ] Classifier-free guidance for controllability

### 📋 Phase 3: 2D Extension
- [ ] Extend to 2D images (MNIST)
- [ ] 2D neural field decoder (coordinate → RGB/grayscale)
- [ ] Sparse image observations (random pixels, patches)
- [ ] Compare to interpolation baselines

### 📋 Phase 4: Advanced Approaches
- [ ] Approach 5: Sparse Observation Conditional Diffusion + NF
- [ ] Approach 6: Temporal Neural Fields (4D for video)
- [ ] Approach 10: Sparse Feature Diffusion + INR
- [ ] Comparative evaluation across approaches

---

## Notebook Structure

Each notebook follows this pattern:

1. **Overview**: Architecture diagram, motivation, goals
2. **Data Generation**: Toy problem setup and visualization
3. **Component Implementation**: Modular building blocks
4. **Model Integration**: Complete end-to-end architecture
5. **Training**: Loss functions, optimization, monitoring
6. **Evaluation**: Visualization, metrics, ablations
7. **Summary**: Results, limitations, next steps

---

## Key Design Decisions

### Why Start with 1D?
- **Debugging**: Easier to visualize and understand
- **Fast iteration**: Quick training and experimentation
- **Validation**: Verify concepts before scaling to 2D/3D
- **Interpretable**: Can directly plot and inspect results

### Why SIREN for Neural Fields?
- **Periodic activations**: Well-suited for signals and natural images
- **Smooth derivatives**: Better continuity properties
- **Proven architecture**: Established in neural field literature
- **Simple and effective**: Good baseline before trying more complex architectures

### Why Latent Diffusion?
- **Efficiency**: 64D latent space vs 256D signal space
- **Semantic**: Latents capture signal structure
- **Fast sampling**: Fewer diffusion steps needed
- **Scalability**: Essential for 2D/3D (can't do pixel-space diffusion on high-res)

### Why Cosine Schedule?
- **Better signal-to-noise**: Improved over linear schedule
- **Standard practice**: Widely used in modern diffusion models
- **Stable training**: Fewer artifacts and better sample quality

---

## Training Tips

### For Best Results:
1. **Encoder-decoder first**: Train without diffusion initially to verify reconstruction
2. **Dual loss balancing**: Start with equal weights (λ_recon = λ_diff = 1.0)
3. **Monitor both losses**: Ensure both reconstruction and diffusion improve
4. **Visual validation**: Check reconstructions every few epochs
5. **Learning rate**: 1e-4 works well, can reduce if unstable

### Common Issues:
- **Poor reconstruction**: Increase λ_recon or train encoder-decoder longer
- **Poor generation**: Increase λ_diff or diffusion timesteps
- **Unstable training**: Reduce learning rate, check for NaN gradients
- **Blurry outputs**: Try different neural field architectures (more layers, higher omega_0)

---

## Experimental Results

### Approach 1 Baseline:
- **Reconstruction MSE**: ~0.001-0.01 (20% sparse observations)
- **Training time**: ~5-10 min (20 epochs, 2000 samples, CPU)
- **Generation quality**: Plausible signals, some diversity
- **Arbitrary resolution**: Smooth across 64 to 1024 points

### Observations:
- Encoder-decoder learns quickly (5-10 epochs sufficient)
- Diffusion requires more epochs (20-30 for good generation)
- Neural field provides excellent continuity
- Latent space is smooth and well-structured

---

## Next Experiments

### Immediate (notebooks 03-04):
1. **Conditional generation**: Notebook 03 - condition diffusion on sparse observations
2. **2D images**: Notebook 04 - extend to MNIST (28×28 → continuous)

### Short-term (notebooks 05-07):
3. **Better encoder**: Transformer for sparse observations
4. **Approach comparison**: Implement Approach 5 or 10
5. **Temporal**: 4D neural field for video (Approach 6)

### Long-term:
6. Real datasets (CIFAR-10, medical imaging)
7. Performance optimization and scaling
8. Publication-ready experiments and baselines

---

## References

### Neural Fields:
- **SIREN**: Implicit Neural Representations with Periodic Activation Functions (Sitzmann et al., 2020)
- **NeRF**: Neural Radiance Fields (Mildenhall et al., 2020)

### Diffusion Models:
- **DDPM**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **Improved DDPM**: Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)
- **Latent Diffusion**: High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)

### Related Work:
- Neural field conditioning and generation
- Sparse signal reconstruction
- Continuous representations for vision

---

## Contact & Contributions

This is an active research project. For questions or suggestions about the implementation, please refer to the main repository README.md.
