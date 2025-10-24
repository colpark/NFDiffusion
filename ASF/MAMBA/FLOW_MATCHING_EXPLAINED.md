# Flow Matching Formulation Explained

## What is Flow Matching?

Flow matching learns a **velocity field** `v(x, t)` that transports noise (at t=0) to data (at t=1) by solving an ODE:

```
dx/dt = v(x, t)
x(0) ~ N(0, I)  (noise)
x(1) ~ p_data   (real image)
```

**Key Question**: What path should `x(t)` take from noise to data?

---

## Current Implementation: Straight-Line Flows

### What We're Doing

**Training**:
```python
def conditional_flow(x_0, x_1, t):
    """Linear interpolation between noise and data"""
    return (1 - t) * x_0 + t * x_1
    #      ^^^^^^^^^^^^^^^^^^^^^^^
    #      STRAIGHT LINE in RGB space

def target_velocity(x_0, x_1):
    """Constant velocity along the straight line"""
    return x_1 - x_0
    #      ^^^^^^^^^
    #      CONSTANT (doesn't depend on t!)
```

**Visualization**:
```
t=0: x_0 (pure noise)
     |
     | straight line
     |
t=0.5: 0.5*x_0 + 0.5*x_1 (halfway)
     |
     | straight line
     |
t=1: x_1 (real image)
```

### Why This Might Be Suboptimal

#### Problem 1: Straight Lines in High Dimensions Are Long

**In 3D RGB space (simplified)**:
```
Noise:    x_0 = [0.8, -0.3, 0.5]  (random)
Image:    x_1 = [0.9,  0.7, 0.2]  (structured)

Straight line: x(t) = (1-t)*x_0 + t*x_1
```

**Issue**: The straight path might go through "unnatural" intermediate states:
```
t=0.3: [0.83, 0.00, 0.41]  <- might not look like any real image!
t=0.5: [0.85, 0.20, 0.35]  <- weird color combination
t=0.7: [0.87, 0.40, 0.29]  <- gradually becoming natural
```

The model has to learn to handle these unnatural intermediate states, which is harder.

---

#### Problem 2: Ignores Data Manifold Structure

**Real images live on a low-dimensional manifold**:
```
All possible RGB values: ℝ^(32×32×3) ≈ 3000 dimensions
Natural images: ~100 dimensional manifold (much smaller!)
```

**Straight lines cut through the manifold**:
```
     Data Manifold
        ___
      /     \
x_0  •       • x_1
      \  ×  /    <- × = straight path goes OUTSIDE manifold!
       \___/

Better path would follow the manifold curvature
```

---

#### Problem 3: Velocity is Not Time-Dependent

**Current velocity**:
```python
v_t = x_1 - x_0  # CONSTANT for all t
```

This means:
- Velocity at t=0.1 (near noise) = velocity at t=0.9 (near data)
- But intuitively, we might want different speeds at different times
- Or different directions as we get closer to the data manifold

---

## Alternative Formulations (Why They Might Be Better)

### 1. **Optimal Transport (OT) Flow**

**Idea**: Find the path that minimizes "transport cost"

```python
# Instead of straight line, solve:
min ∫ ||dx/dt||² dt
subject to: x(0) = x_0, x(1) = x_1
```

**Result**: Geodesic paths that follow the natural geometry

**Example**:
```
Straight-line flow:
x_0 →→→→→→→→→→ x_1  (distance = 10, many unnatural states)

OT flow:
x_0 →→↓↓↓→→→ x_1  (distance = 12, but stays on manifold!)
      ↓
   (curves along natural image space)
```

**Advantages**:
- Shorter paths in the right metric
- Stays closer to natural images during transport
- **Less noise** because intermediate states are more "real"

**Implementation**:
```python
# Requires solving Sinkhorn iterations during training
from ott.solvers import sinkhorn
cost_matrix = compute_cost(x_0, x_1)
ot_plan = sinkhorn(cost_matrix)
# Then sample paths from OT plan
```

---

### 2. **Rectified Flow**

**Idea**: Train a flow, then "straighten" it iteratively

**Algorithm**:
1. Train initial flow (can be suboptimal)
2. Sample pairs (x_0, x_1) from the learned flow
3. Retrain on straight lines between these samples
4. Repeat → paths become straighter and straighter

**Why it works**:
```
Iteration 1: Learned flow is curved (follows manifold)
  x_0 ~~curved~~> x_1

Iteration 2: Straighten those learned paths
  x_0 →→→→→→→ x_1  (but now the straight line is GOOD
                     because endpoints came from manifold)

Iteration 3: Even straighter
  x_0 ————————> x_1  (nearly optimal straight lines)
```

**Result**: Fast, straight flows that still respect geometry

---

### 3. **Conditional Flow Matching (CFM)**

**Idea**: Learn time-dependent velocity fields

```python
# Instead of constant velocity
v_t = x_1 - x_0  # current (constant)

# Use learned time-dependent velocity
v_t = f(x_t, t, x_1)  # depends on current position AND time
```

**Example trajectory**:
```
t=0.0: v = x_1 - x_0         (start moving toward data)
t=0.5: v = adjust_for_manifold(x_0.5, x_1)
t=0.9: v = fine_tune(x_0.9, x_1)  (slow down, refine)
```

---

## How This Relates to Your Noise Problem

### Current Situation

**Your observation**: Images are noisy/wiggly at all scales

**Possible connection to straight-line flows**:

1. **Unnatural Intermediate States**:
   ```python
   # During sampling, we integrate the ODE
   x_0 = noise
   for t in [0, 0.02, 0.04, ..., 1.0]:  # 50 steps
       v = model(x_t, t)  # predict velocity
       x_{t+dt} = x_t + dt * v
   ```

   If the model learned on straight-line paths that go through unnatural states:
   - Model might be uncertain at intermediate times
   - Uncertainty → noisy predictions → accumulated error → final noise

2. **Velocity Field Is Harder to Learn**:
   ```python
   # Model must learn v such that:
   x(0) = noise  →  x(0.1) = ?  →  x(0.2) = ?  → ... → x(1) = image
   ```

   With straight lines:
   - x(0.1) might look very unnatural
   - Model struggles to predict correct v at unnatural states
   - Errors compound → noise in final output

3. **Off-Manifold Regions**:
   If straight line goes outside natural image manifold:
   ```
                  Data Manifold
                     ____
   x_0 (noise)    /      \
        •        /        \  x_1 (image)
         \      •  x_0.5   •
          \    /    ^      /
           \  /     |     /
            \/      |    /
                 outside!
   ```
   Model is asked to predict velocities in regions it never saw during training on real images → unstable predictions → noise

---

## Why This Might NOT Be the Main Issue (40% unlikely)

### Counterarguments:

1. **Straight-line flows work well in practice**:
   - Flow Matching paper shows good results with straight lines
   - Many successful applications use this simple formulation
   - If network is expressive enough, it can learn to handle it

2. **Your noise is consistent across scales**:
   - OT/rectified flow would improve quality, but noise pattern suggests:
     - Insufficient ODE steps (more likely)
     - Training convergence (more likely)
   - Because OT would affect *structure*, not just *noise level*

3. **Implementation complexity**:
   - OT requires Sinkhorn iterations → slower training
   - Rectified flow requires multiple training rounds
   - Might not be worth the effort if simpler fixes work

---

## How to Test If This Is the Problem

### Diagnostic 1: Check Intermediate States

```python
@torch.no_grad()
def visualize_flow_path(model, x_0, x_1):
    """Visualize the learned flow from noise to image"""

    t_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    for i, t in enumerate(t_values):
        # Straight-line interpolation
        x_straight = (1 - t) * x_0 + t * x_1
        axes[0, i].imshow(to_image(x_straight))
        axes[0, i].set_title(f't={t:.1f} (straight)')

        # Learned flow (integrate ODE)
        x_learned = integrate_ode(model, x_0, t_end=t)
        axes[1, i].imshow(to_image(x_learned))
        axes[1, i].set_title(f't={t:.1f} (learned)')

    plt.suptitle('Straight-line vs Learned Flow')
    plt.show()
```

**What to look for**:
- If learned flow deviates significantly from straight line → OT might help
- If learned flow follows straight line closely → not the issue

---

### Diagnostic 2: Measure Path Length

```python
def compute_path_length(model, x_0, x_1, num_steps=100):
    """Compute total path length of learned flow"""

    x = x_0.clone()
    total_length = 0.0

    for t in np.linspace(0, 1, num_steps):
        v = model(x, t)  # velocity
        total_length += torch.norm(v).item() / num_steps
        x = x + v / num_steps  # Euler step

    straight_length = torch.norm(x_1 - x_0).item()

    print(f"Learned path length: {total_length:.4f}")
    print(f"Straight path length: {straight_length:.4f}")
    print(f"Ratio: {total_length / straight_length:.4f}")

    # If ratio >> 1, learned flow is very curved
    # If ratio ≈ 1, learned flow is nearly straight
```

**Interpretation**:
- Ratio > 1.5: Flow is very curved → OT might improve
- Ratio ≈ 1.0: Flow already straight → not the issue

---

### Diagnostic 3: Compare OT vs Straight-Line

```python
# Train two models side-by-side
model_straight = MAMBADiffusion()  # current
model_ot = MAMBADiffusionOT()      # with OT

# Train both
train(model_straight, use_ot=False)
train(model_ot, use_ot=True)

# Compare quality
quality_straight = evaluate(model_straight)
quality_ot = evaluate(model_ot)

print(f"Straight-line PSNR: {quality_straight:.2f} dB")
print(f"OT PSNR: {quality_ot:.2f} dB")
```

---

## Quick Summary

**What "straight-line may be suboptimal" means**:

1. **Current approach**: Interpolate linearly between noise and data
   - Simple, fast, easy to implement
   - But might go through "unnatural" intermediate states

2. **Alternative (OT)**: Follow geodesic paths that respect data geometry
   - More natural intermediate states
   - Potentially less noise in final output
   - But more complex to implement

3. **Why it's only 60% likely to be an issue**:
   - Straight-line flows work well in many cases
   - Your noise pattern suggests simpler issues (ODE steps, training)
   - Worth investigating if simple fixes don't work

4. **How to test**:
   - Visualize learned flow vs straight-line
   - Measure path length ratio
   - Train with OT and compare (if needed)

---

## Recommendation

**Priority**: Test this AFTER trying simpler fixes:

1. ✅ First: Increase ODE steps (90% likely to help)
2. ✅ Second: Train longer (80% likely to help)
3. ✅ Third: Adjust Fourier scale (75% likely to help)
4. ⏸️ Fourth: Try OT flows (60% likely to help, but complex)

If steps 1-3 don't fix the noise, then OT is worth exploring.
