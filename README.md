# NFDiffusion: Neural Field Diffusion Models

**Goal**: Build a continuity-aware neural architecture that can perform arbitrary-scale upsampling by merging neural fields with diffusion models.

## Project Vision

### Core Capabilities
- **Continuity-Aware**: Operator that understands smooth transitions and preserves continuity
- **Arbitrary-Scale Upsampling**: Generate high-resolution outputs at any scale
- **Sparse Data Handling**: Works with sparse inputs and outputs
- **Multiple Sparsity Types**: Temporal, spatial, feature-level sparsity

### Why Neural Fields + Diffusion?

**Neural Fields**:
- Continuous representation (infinite resolution)
- Coordinate-based queries (arbitrary sampling)
- Implicit smoothness and continuity

**Diffusion Models**:
- Powerful generative modeling
- High-quality synthesis
- Flexible conditioning

**Combined**: Continuous, generative, multi-scale operator with sparse data handling

## Repository Structure

```
NFDiffusion/
├── README.md                    # This file
├── APPROACHES.md                # Detailed approaches for merging NF + Diffusion
├── research/                    # Research notes and papers
├── experiments/                 # Experimental implementations
├── core/                        # Core library code
│   ├── neural_fields/          # Neural field implementations
│   ├── diffusion/              # Diffusion model components
│   └── sparse/                 # Sparse data handling
└── notebooks/                   # Jupyter notebooks for exploration
```

## Quick Start

*(Coming soon)*

## References

*(Coming soon)*
