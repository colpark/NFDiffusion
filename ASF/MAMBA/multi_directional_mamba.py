"""
Multi-Directional MAMBA Implementation

Addresses spatial locality problem in MAMBA by scanning the 2D coordinate space
in multiple directions and fusing the results.

Key Innovation:
- MAMBA is 1D sequential (loses 2D spatial structure)
- Solution: Process 4 different orderings, merge results
- Captures horizontal, vertical, and diagonal relationships

Author: Claude Code
Date: 2025-10-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def order_by_row(coords):
    """
    Row-major ordering (left to right, top to bottom)

    Args:
        coords: (B, N, 2) where coords[..., 0]=x, coords[..., 1]=y

    Returns:
        indices: (B, N) ordering indices
    """
    B, N, _ = coords.shape
    indices_list = []

    for b in range(B):
        # Sort by y first (row), then by x (column)
        y_vals = coords[b, :, 1]
        x_vals = coords[b, :, 0]

        # Create composite sort key: y * large_number + x
        sort_keys = y_vals * 1000 + x_vals
        indices = torch.argsort(sort_keys)
        indices_list.append(indices)

    return torch.stack(indices_list, dim=0)


def order_by_column(coords):
    """
    Column-major ordering (top to bottom, left to right)

    Args:
        coords: (B, N, 2) where coords[..., 0]=x, coords[..., 1]=y

    Returns:
        indices: (B, N) ordering indices
    """
    B, N, _ = coords.shape
    indices_list = []

    for b in range(B):
        # Sort by x first (column), then by y (row)
        y_vals = coords[b, :, 1]
        x_vals = coords[b, :, 0]

        # Create composite sort key: x * large_number + y
        sort_keys = x_vals * 1000 + y_vals
        indices = torch.argsort(sort_keys)
        indices_list.append(indices)

    return torch.stack(indices_list, dim=0)


def order_by_diagonal(coords):
    """
    Diagonal ordering (top-left to bottom-right)
    Points along same diagonal (x+y=const) are grouped together

    Args:
        coords: (B, N, 2) where coords[..., 0]=x, coords[..., 1]=y

    Returns:
        indices: (B, N) ordering indices
    """
    B, N, _ = coords.shape
    indices_list = []

    for b in range(B):
        y_vals = coords[b, :, 1]
        x_vals = coords[b, :, 0]

        # Diagonal: constant sum x+y
        # Within diagonal: order by x (or y, doesn't matter)
        diag_vals = x_vals + y_vals
        sort_keys = diag_vals * 1000 + x_vals
        indices = torch.argsort(sort_keys)
        indices_list.append(indices)

    return torch.stack(indices_list, dim=0)


def order_by_antidiagonal(coords):
    """
    Anti-diagonal ordering (top-right to bottom-left)
    Points along same anti-diagonal (x-y=const) are grouped together

    Args:
        coords: (B, N, 2) where coords[..., 0]=x, coords[..., 1]=y

    Returns:
        indices: (B, N) ordering indices
    """
    B, N, _ = coords.shape
    indices_list = []

    for b in range(B):
        y_vals = coords[b, :, 1]
        x_vals = coords[b, :, 0]

        # Anti-diagonal: constant difference x-y
        # Within anti-diagonal: order by x
        antidiag_vals = x_vals - y_vals
        sort_keys = antidiag_vals * 1000 + x_vals
        indices = torch.argsort(sort_keys)
        indices_list.append(indices)

    return torch.stack(indices_list, dim=0)


def reorder_sequence(x, indices):
    """
    Reorder sequence according to indices

    Args:
        x: (B, N, D) input sequence
        indices: (B, N) reordering indices

    Returns:
        x_reordered: (B, N, D) reordered sequence
    """
    B, N, D = x.shape

    # Expand indices for gathering
    indices_expanded = indices.unsqueeze(-1).expand(B, N, D)

    # Gather
    x_reordered = torch.gather(x, dim=1, index=indices_expanded)

    return x_reordered


def inverse_reorder(x, indices):
    """
    Reverse the reordering (scatter back to original positions)

    Args:
        x: (B, N, D) reordered sequence
        indices: (B, N) the indices used for original reordering

    Returns:
        x_original: (B, N, D) sequence in original order
    """
    B, N, D = x.shape

    # Create inverse indices
    inverse_indices = torch.zeros_like(indices)
    for b in range(B):
        inverse_indices[b, indices[b]] = torch.arange(N, device=indices.device)

    # Scatter back
    indices_expanded = inverse_indices.unsqueeze(-1).expand(B, N, D)
    x_original = torch.gather(x, dim=1, index=indices_expanded)

    return x_original


class MultiDirectionalSSM(nn.Module):
    """
    Multi-directional State Space Model

    Processes input sequence in 4 directions:
    1. Horizontal (row-wise)
    2. Vertical (column-wise)
    3. Diagonal (top-left to bottom-right)
    4. Anti-diagonal (top-right to bottom-left)

    Then fuses the outputs to capture full 2D spatial structure.
    """
    def __init__(self, ssm_block_class, d_model, d_state=16, dropout=0.1):
        """
        Args:
            ssm_block_class: Class to use for SSM (e.g., SSMBlockFast)
            d_model: Model dimension
            d_state: SSM state dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model

        # Create 4 separate SSM blocks for each direction
        self.ssm_horizontal = ssm_block_class(d_model, d_state, dropout)
        self.ssm_vertical = ssm_block_class(d_model, d_state, dropout)
        self.ssm_diagonal = ssm_block_class(d_model, d_state, dropout)
        self.ssm_antidiagonal = ssm_block_class(d_model, d_state, dropout)

        # Fusion: combine 4 directional outputs
        # Option 1: Concatenate and project
        self.fusion = nn.Sequential(
            nn.Linear(4 * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        # Option 2: Learnable weighted average
        self.direction_weights = nn.Parameter(torch.ones(4) / 4)

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, coords):
        """
        Args:
            x: (B, N, d_model) input features
            coords: (B, N, 2) coordinates for ordering

        Returns:
            y: (B, N, d_model) output features with 2D spatial awareness
        """
        B, N, D = x.shape

        # Get ordering for each direction
        indices_h = order_by_row(coords)
        indices_v = order_by_column(coords)
        indices_d = order_by_diagonal(coords)
        indices_a = order_by_antidiagonal(coords)

        # 1. Horizontal pass
        x_h = reorder_sequence(x, indices_h)
        y_h = self.ssm_horizontal(x_h)
        y_h = inverse_reorder(y_h, indices_h)

        # 2. Vertical pass
        x_v = reorder_sequence(x, indices_v)
        y_v = self.ssm_vertical(x_v)
        y_v = inverse_reorder(y_v, indices_v)

        # 3. Diagonal pass
        x_d = reorder_sequence(x, indices_d)
        y_d = self.ssm_diagonal(x_d)
        y_d = inverse_reorder(y_d, indices_d)

        # 4. Anti-diagonal pass
        x_a = reorder_sequence(x, indices_a)
        y_a = self.ssm_antidiagonal(x_a)
        y_a = inverse_reorder(y_a, indices_a)

        # Fusion strategy 1: Concatenate and project (default)
        y_concat = torch.cat([y_h, y_v, y_d, y_a], dim=-1)  # (B, N, 4*d_model)
        y_fused = self.fusion(y_concat)  # (B, N, d_model)

        # Fusion strategy 2: Weighted average (alternative, commented out)
        # weights = F.softmax(self.direction_weights, dim=0)
        # y_fused = weights[0] * y_h + weights[1] * y_v + weights[2] * y_d + weights[3] * y_a

        # Residual connection and normalization
        y = x + y_fused
        y = self.norm(y)

        return y


class MultiDirectionalMambaBlock(nn.Module):
    """
    Complete multi-directional Mamba block

    Replaces standard MambaBlock with multi-directional version
    """
    def __init__(self, ssm_block_class, d_model, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()

        # Expand
        self.proj_in = nn.Linear(d_model, d_model * expand_factor)

        # Multi-directional SSM
        self.multi_ssm = MultiDirectionalSSM(
            ssm_block_class,
            d_model * expand_factor,
            d_state,
            dropout
        )

        # Contract
        self.proj_out = nn.Linear(d_model * expand_factor, d_model)

        # MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, coords):
        """
        Args:
            x: (B, N, d_model) input features
            coords: (B, N, 2) spatial coordinates
        """
        # SSM branch with multi-directional processing
        residual = x
        x = self.proj_in(x)
        x = self.multi_ssm(x, coords)
        x = self.proj_out(x)
        x = x + residual

        # MLP branch
        x = x + self.mlp(x)

        return x


# Test the multi-directional ordering
if __name__ == "__main__":
    print("Testing Multi-Directional Ordering Functions\n")

    # Create a simple 4x4 grid
    size = 4
    y, x = torch.meshgrid(
        torch.linspace(0, 1, size),
        torch.linspace(0, 1, size),
        indexing='ij'
    )
    coords = torch.stack([x.flatten(), y.flatten()], dim=-1).unsqueeze(0)  # (1, 16, 2)

    print(f"Test grid: {size}x{size} = {size**2} points")
    print(f"Coords shape: {coords.shape}\n")

    # Test each ordering
    indices_h = order_by_row(coords)
    indices_v = order_by_column(coords)
    indices_d = order_by_diagonal(coords)
    indices_a = order_by_antidiagonal(coords)

    print("Row-wise ordering (horizontal):")
    print(indices_h[0].reshape(size, size))
    print()

    print("Column-wise ordering (vertical):")
    print(indices_v[0].reshape(size, size))
    print()

    print("Diagonal ordering:")
    print(indices_d[0].reshape(size, size))
    print()

    print("Anti-diagonal ordering:")
    print(indices_a[0].reshape(size, size))
    print()

    # Test reordering and inverse
    x_test = torch.randn(1, 16, 8)
    x_reordered = reorder_sequence(x_test, indices_h)
    x_recovered = inverse_reorder(x_reordered, indices_h)

    print(f"Reordering test - Max error: {(x_test - x_recovered).abs().max().item():.6f}")
    print("✓ Ordering functions working correctly!" if (x_test - x_recovered).abs().max() < 1e-6 else "✗ Error in ordering!")
