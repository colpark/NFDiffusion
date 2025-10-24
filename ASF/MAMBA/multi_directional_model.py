"""
Multi-Directional MAMBA Diffusion Model

Complete implementation with multi-directional scanning to address
spatial locality issues in MAMBA.

Usage:
    Replace MAMBADiffusion with MultiDirectionalMAMBADiffusion in training notebook

Author: Claude Code
Date: 2025-10-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import the multi-directional components
from multi_directional_mamba import (
    MultiDirectionalMambaBlock,
    MultiDirectionalSSM,
    order_by_row,
    order_by_column,
    order_by_diagonal,
    order_by_antidiagonal
)

# Import original components (you'll need these from your notebook)
# Assuming SSMBlockFast and other components are available
# If not, you'll need to import them from the notebook


class SinusoidalTimeEmbedding(nn.Module):
    """Time embedding for diffusion"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class MultiDirectionalMAMBADiffusion(nn.Module):
    """
    Multi-Directional State Space Model for Sparse Field Diffusion

    Key Innovation:
    - Processes coordinates in 4 directions (horizontal, vertical, diagonal, anti-diagonal)
    - Fuses directional outputs to capture full 2D spatial structure
    - Addresses MAMBA's limitation of 1D sequential processing

    Args:
        num_fourier_feats: Number of Fourier feature frequencies
        d_model: Model dimension
        num_layers: Number of MAMBA layers
        d_state: SSM state dimension
        dropout: Dropout rate
        ssm_block_class: Which SSM implementation to use (e.g., SSMBlockFast)
    """
    def __init__(
        self,
        ssm_block_class,  # You need to pass this!
        num_fourier_feats=256,
        d_model=512,
        num_layers=6,
        d_state=16,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Fourier features (import FourierFeatures from your code)
        # You'll need to import this from core.neural_fields.perceiver
        from core.neural_fields.perceiver import FourierFeatures
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
        # KEY CHANGE: Multi-directional MAMBA blocks
        # ============================================
        self.mamba_blocks = nn.ModuleList([
            MultiDirectionalMambaBlock(
                ssm_block_class,  # Pass SSMBlockFast or SSMBlock
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
        Forward pass with multi-directional spatial awareness

        Args:
            noisy_values: (B, N_out, 3) noisy RGB at query points
            query_coords: (B, N_out, 2) query coordinates
            t: (B,) timestep
            input_coords: (B, N_in, 2) sparse input coordinates
            input_values: (B, N_in, 3) sparse input RGB values

        Returns:
            (B, N_out, 3) predicted velocity
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
        )  # (B, N_in, d_model)

        query_tokens = self.query_proj(
            torch.cat([query_feats, noisy_values], dim=-1)
        )  # (B, N_out, d_model)

        # Add time embedding
        input_tokens = input_tokens + t_emb.unsqueeze(1)
        query_tokens = query_tokens + t_emb.unsqueeze(1)

        # ============================================
        # KEY CHANGE: Concatenate coordinates for multi-directional processing
        # ============================================
        all_coords = torch.cat([input_coords, query_coords], dim=1)  # (B, N_in+N_out, 2)
        seq = torch.cat([input_tokens, query_tokens], dim=1)  # (B, N_in+N_out, d_model)

        # ============================================
        # Process through multi-directional MAMBA blocks
        # Each block scans in 4 directions and fuses results
        # ============================================
        for mamba_block in self.mamba_blocks:
            seq = mamba_block(seq, all_coords)  # ← Pass coords for ordering!

        # Split back into input and query sequences
        input_seq = seq[:, :N_in, :]  # (B, N_in, d_model)
        query_seq = seq[:, N_in:, :]  # (B, N_out, d_model)

        # Cross-attention: queries attend to processed inputs
        output, _ = self.query_cross_attn(query_seq, input_seq, input_seq)

        # Decode to RGB velocity
        return self.decoder(output)


# ============================================
# Lightweight Version (Shared Weights)
# ============================================

class LightweightMultiDirectionalMAMBADiffusion(nn.Module):
    """
    Lightweight version with shared SSM weights across directions

    Advantages:
    - Same parameter count as original MAMBA
    - Still gets multi-directional information
    - Faster training than full multi-directional

    Disadvantages:
    - Shared weights may learn "average" behavior
    - Potentially less powerful than dedicated direction weights
    """
    def __init__(
        self,
        ssm_block_class,
        num_fourier_feats=256,
        d_model=512,
        num_layers=6,
        d_state=16,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Same as full version
        from core.neural_fields.perceiver import FourierFeatures
        self.fourier = FourierFeatures(coord_dim=2, num_freqs=num_fourier_feats, scale=10.0)
        feat_dim = num_fourier_feats * 2

        self.input_proj = nn.Linear(feat_dim + 3, d_model)
        self.query_proj = nn.Linear(feat_dim + 3, d_model)

        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Lightweight multi-directional blocks (shared weights)
        self.mamba_blocks = nn.ModuleList([
            LightweightMultiDirectionalBlock(
                ssm_block_class,
                d_model,
                d_state,
                expand_factor=2,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.query_cross_attn = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=dropout, batch_first=True
        )

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
        # Same as full version
        B = query_coords.shape[0]
        N_in = input_coords.shape[1]
        N_out = query_coords.shape[1]

        t_emb = self.time_mlp(self.time_embed(t))

        input_feats = self.fourier(input_coords)
        query_feats = self.fourier(query_coords)

        input_tokens = self.input_proj(
            torch.cat([input_feats, input_values], dim=-1)
        )
        query_tokens = self.query_proj(
            torch.cat([query_feats, noisy_values], dim=-1)
        )

        input_tokens = input_tokens + t_emb.unsqueeze(1)
        query_tokens = query_tokens + t_emb.unsqueeze(1)

        all_coords = torch.cat([input_coords, query_coords], dim=1)
        seq = torch.cat([input_tokens, query_tokens], dim=1)

        for mamba_block in self.mamba_blocks:
            seq = mamba_block(seq, all_coords)

        input_seq = seq[:, :N_in, :]
        query_seq = seq[:, N_in:, :]

        output, _ = self.query_cross_attn(query_seq, input_seq, input_seq)

        return self.decoder(output)


class LightweightMultiDirectionalBlock(nn.Module):
    """Lightweight multi-directional block with shared SSM weights"""
    def __init__(self, ssm_block_class, d_model, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()

        self.proj_in = nn.Linear(d_model, d_model * expand_factor)

        # SINGLE SSM shared across all directions
        self.ssm = ssm_block_class(d_model * expand_factor, d_state, dropout)

        # Lightweight fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * expand_factor * 4, d_model * expand_factor * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand_factor * 2, d_model * expand_factor)
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
        from multi_directional_mamba import (
            order_by_row, order_by_column, order_by_diagonal, order_by_antidiagonal,
            reorder_sequence, inverse_reorder
        )

        residual = x
        x = self.proj_in(x)

        # Get 4 orderings
        indices = [
            order_by_row(coords),
            order_by_column(coords),
            order_by_diagonal(coords),
            order_by_antidiagonal(coords)
        ]

        # Process in each direction with SAME SSM
        outputs = []
        for idx in indices:
            x_ordered = reorder_sequence(x, idx)
            y = self.ssm(x_ordered)
            y = inverse_reorder(y, idx)
            outputs.append(y)

        # Fuse all 4 outputs
        y_fused = self.fusion(torch.cat(outputs, dim=-1))

        x = self.proj_out(y_fused)
        x = x + residual

        # MLP
        x = x + self.mlp(x)

        return x


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    print("Multi-Directional MAMBA Diffusion Model")
    print("=" * 60)

    # You need to import SSMBlockFast from your notebook
    # For this example, we'll show the structure

    print("\n1. Full Multi-Directional Version:")
    print("   - 4 separate SSM blocks per layer per direction")
    print("   - 4× more parameters in SSM layers")
    print("   - Maximum spatial awareness")
    print("   - Recommended for final model")

    print("\n2. Lightweight Version:")
    print("   - Single SSM shared across 4 directions")
    print("   - Same parameters as original")
    print("   - Good for quick testing")
    print("   - Trade-off: Shared weights may learn average behavior")

    print("\n" + "=" * 60)
    print("Integration Steps:")
    print("1. Import: from MAMBA.multi_directional_model import MultiDirectionalMAMBADiffusion")
    print("2. Replace: model = MultiDirectionalMAMBADiffusion(SSMBlockFast, ...)")
    print("3. Train: Same training loop, no other changes needed!")
    print("=" * 60)
