# MAMBA SSM Numerical Stability Fixes

## Problem: NaN Values During Training

The original MAMBA implementation suffered from numerical instability causing NaN values during training.

## Root Causes Identified

### 1. Missing Proper Discretization
**Original Code (WRONG)**:
```python
h = A.unsqueeze(0) * h + self.B(x_t)
```

**Problem**: This is not a valid discretization of the continuous-time SSM. It's missing the time step `dt` and proper zero-order hold discretization.

**Correct Discretization**:
```python
dt = 1.0 / N  # Time step
A_discrete = exp(dt * A)  # Matrix exponential
B_discrete = (A_discrete - I) / A  # Proper scaling
h = A_discrete * h + B_discrete * B(x)
```

### 2. Division by Zero Risk
**Problem**: When computing `B_discrete = (A_discrete - 1) / A`, if `A ≈ 0`, this causes instability.

**Fix**: Safe division with torch.where:
```python
B_discrete_scale = torch.where(
    torch.abs(A) > eps,
    (A_discrete - 1.0) / (A + eps),
    torch.ones_like(A) * dt  # Limit case when A→0
)
```

### 3. State Explosion Over Long Sequences
**Problem**: With sequence length N=408 (204 inputs + 204 queries), the state `h` can explode as it accumulates over 408 steps.

**Fix**: Clamp state values:
```python
h = torch.clamp(h, min=-10.0, max=10.0)
```

### 4. Poor Parameter Initialization
**Original**:
```python
self.A_log = nn.Parameter(torch.randn(d_state))  # Random ~N(0,1)
self.D = nn.Parameter(torch.randn(d_model))      # Random ~N(0,1)
```

**Problem**: Random initialization can start with extreme values, causing immediate instability.

**Fix**: Carefully initialized parameters:
```python
# A_log initialized near -1, so A ≈ -exp(-1) ≈ -0.37 (stable decay)
self.A_log = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)

# D initialized very small (skip connection shouldn't dominate)
self.D = nn.Parameter(torch.randn(d_model) * 0.01)

# B and C with smaller gains
nn.init.xavier_uniform_(self.B.weight, gain=0.5)
nn.init.xavier_uniform_(self.C.weight, gain=0.5)
```

### 5. Unbounded A Matrix
**Original**:
```python
A = -torch.exp(self.A_log)  # No bounds
```

**Problem**: If `A_log` becomes very large during training, `A` can explode.

**Fix**: Clamp A values:
```python
A = -torch.exp(self.A_log).clamp(min=eps, max=10.0)
```

## Complete Fixed SSMBlock

```python
class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Initialize A_log near -1 for stability
        self.A_log = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)

        # Input-to-state and state-to-output projections
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)

        # Small skip connection
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

        # Careful weight initialization
        nn.init.xavier_uniform_(self.B.weight, gain=0.5)
        nn.init.xavier_uniform_(self.C.weight, gain=0.5)

        # Gating for selective updates
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8

    def forward(self, x):
        B, N, D = x.shape

        # Get A matrix (negative, clamped for stability)
        A = -torch.exp(self.A_log).clamp(min=self.eps, max=10.0)

        # Discretization timestep
        dt = 1.0 / N

        # Zero-order hold discretization
        A_discrete = torch.exp(dt * A)

        # Safe B discretization
        B_discrete_scale = torch.where(
            torch.abs(A) > self.eps,
            (A_discrete - 1.0) / (A + self.eps),
            torch.ones_like(A) * dt
        )

        # Initialize state
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(N):
            x_t = x[:, t, :]
            B_x = self.B(x_t)

            # State update with proper discretization
            h = A_discrete.unsqueeze(0) * h + B_discrete_scale.unsqueeze(0) * B_x

            # Prevent state explosion
            h = torch.clamp(h, min=-10.0, max=10.0)

            # Compute output
            y_t = self.C(h) + self.D * x_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        # Gating and residual
        gate = self.gate(x)
        y = gate * y + (1 - gate) * x

        return self.dropout(self.norm(y))
```

## Mathematical Background

### Continuous-Time SSM
```
h'(t) = A h(t) + B x(t)
y(t) = C h(t) + D x(t)
```

### Zero-Order Hold Discretization
For time step `dt`:
```
h_k = exp(dt * A) h_{k-1} + [(exp(dt * A) - I) A^{-1}] B x_k
    = A_discrete h_{k-1} + B_discrete B x_k
```

where:
- `A_discrete = exp(dt * A)`
- `B_discrete = (A_discrete - I) / A`

### Numerical Stability Considerations

1. **Exponential Stability**: A must be negative for stable decay
2. **Division Safety**: Handle A → 0 case with limit: `lim_{A→0} (exp(dt*A) - 1)/A = dt`
3. **State Bounds**: Prevent explosion with clamping
4. **Parameter Initialization**: Start in stable regime

## Expected Behavior After Fixes

✅ **No NaN values** during training
✅ **Stable gradient flow** through all 6 SSM layers
✅ **Consistent loss** decrease over epochs
✅ **Successful training** to convergence

## Testing

After applying fixes:
1. Run model test (cell 7) - should complete without errors
2. Train for 1 epoch - check for NaN in loss
3. Monitor state values (should stay bounded)
4. Verify gradients (should be finite)

## Performance Impact

The fixes have minimal performance impact:
- **Speed**: ~same (added clamps are cheap)
- **Memory**: ~same (no additional buffers)
- **Quality**: **Better** (stable training enables convergence)
