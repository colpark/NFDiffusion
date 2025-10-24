# Stabilizing Sparse Flow Matching on CIFAR‑10 with Mamba/SSM  
*A practical playbook for denoising + continuous-field reconstruction that also super-resolves*

> **Context.** Flow-matching model trained on CIFAR‑10 at 32×32; inputs are ~20% sparse points, outputs are 20% sparse points.  
> Observation: reconstructions at the native 32×32 look noisy and don't converge well, but sampling the learned field on denser grids (64×64, 96×96) looks much better—suggesting scale/ordering/aliasing mismatches rather than model capacity limits.

![Native vs multi-scale reconstructions](mamba_multiscale.png)
![Sparse input and reconstruction at 32×32](mamba_sparse_32.png)

---

## TL;DR – What to change first

1. **Use pixel-center coordinates** everywhere (dataset, training, visualization).  
   → avoids half‑pixel misalignment that bites at 32×32 and is masked at higher resolutions.
2. **Give the SSM a geometric order and timescale**: sort tokens along a space‑filling curve and set `Δt` from *Euclidean step length* rather than `1/N`.  
   → removes sequence-length dependence and reduces noise at the native grid.
3. **Band‑limit positional encodings** to the grid’s Nyquist frequency (or make them resolution‑aware).  
   → prevents high‑freq aliasing that shows up as pepper/checker noise at 32×32.
4. **Make cross‑attention local** (KNN) and **use relative offsets**. Optionally predict a residual on top of an RBF interpolation prior.  
   → stabilizes learning and improves color/edge faithfulness.
5. **Training hygiene**: normalize to `[-1,1]`, use EMA, curriculum over query density, avoid dropout inside the SSM, and use a slightly higher LR with warmup.

Implementing **(1)** and **(2)** typically fixes the 32×32 noise immediately while keeping the nice 64/96 super‑res behavior.

---

## 1) Coordinate system: sample **pixel centers**, not edges

If you build your 32×32 grid with `linspace(0,1,32)`, you include the edges `0` and `1`. If your sparse points are sampled at pixel centers (common), that creates a half‑pixel offset at native resolution.

**Use this helper everywhere (dataset, training viz, and inference):**

```python
def make_center_grid(size, device):
    # centers in [0,1]: 0.5/size, 1.5/size, ..., (size-0.5)/size
    p = torch.linspace(0.5/size, 1 - 0.5/size, size, device=device)
    y, x = torch.meshgrid(p, p, indexing='ij')
    return torch.stack([x.reshape(-1), y.reshape(-1)], -1)
```

> Also generate multi-scale grids (64/96/…) with **the same rule** so the field is queried consistently across scales.

---

## 2) Make the SSM **geometry‑aware** and **sequence‑length invariant**

### 2.1 Sort tokens along a space‑filling curve

Mamba/SSM assumes a meaningful 1‑D order. Concatenating input/query tokens in arbitrary order creates discontinuities. Sort the *combined* coordinate set along a Morton (Z‑order) or Hilbert curve, run the SSM, then unpermute.

```python
def morton_order(coords):
    # coords in [0,1], returns argsort indices along Morton order
    xy = (coords.clamp(0, 1) * 65535).long()    # (B,N,2)
    x, y = xy[..., 0], xy[..., 1]

    def part1by1(v):
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    code = (part1by1(x) << 1) | part1by1(y)     # (B,N)
    return torch.argsort(code, dim=1)           # (B,N)
```

**In your forward pass** (simplified):

```python
# concatenate
all_coords  = torch.cat([input_coords, query_coords], dim=1)   # (B,N_tot,2)
perm        = morton_order(all_coords)                         # (B,N_tot)
inv_perm    = torch.argsort(perm, dim=1)

seq = torch.cat([input_tokens, query_tokens], dim=1)           # (B,N_tot,D)
seq = torch.gather(seq, 1, perm.unsqueeze(-1).expand(-1, -1, seq.size(-1)))
```

Run the SSM on `seq`, then unpermute back before cross‑attention/decoding:

```python
seq = torch.gather(seq, 1, inv_perm.unsqueeze(-1).expand(-1, -1, seq.size(-1)))
input_seq, query_seq = seq[:, :N_in, :], seq[:, N_in:, :]
```

### 2.2 Use **per‑step Δt** from geometric step lengths

Your SSM sets `dt = 1/N`, so its dynamics depend on how many tokens you have. At 64/96 you use more tokens → smaller steps → smoother behavior → *appears* better. Fix by deriving `Δt` from coordinate distance.

**Compute Δt along the sorted path:**

```python
# after sorting all_coords
diffs = all_coords[:, 1:, :] - all_coords[:, :-1, :]
ds    = torch.norm(diffs, dim=-1)                     # (B,N_tot-1)
ds    = torch.nn.functional.pad(ds, (1,0), value=0.0) # (B,N_tot) first step 0
# normalize scale per batch (optional but helpful)
dt    = ds / (ds.mean(dim=1, keepdim=True) + 1e-8)    # (B,N_tot)
```

**Modify the SSM to accept per‑token `dt`:**

```python
class SSMBlockFast(nn.Module):
    # ...
    def forward(self, x, dt=None):
        B, N, D = x.shape
        A = -torch.exp(self.A_log).clamp(min=1e-8, max=10.0)   # (d_state,)
        Bu = self.B(x)

        if dt is None:
            dt = torch.ones(N, device=x.device, dtype=x.dtype)
        if dt.ndim == 1:          # (N,)
            dt = dt.unsqueeze(0).expand(B, -1)  # (B,N)

        # compute cumulative times for each batch
        t = torch.cumsum(dt, dim=1)                              # (B,N)
        # decay[i,j] = exp(A * (t_i - t_j)) if i>=j else 0
        idx = torch.arange(N, device=x.device)
        mask = (idx.unsqueeze(0) >= idx.unsqueeze(1)).float()    # (N,N)

        diff = (t.unsqueeze(2) - t.unsqueeze(1)).clamp(min=0.0)  # (B,N,N)
        decay = torch.exp(diff.unsqueeze(-1) * A.view(1,1,1,-1)) # (B,N,N,d_state)
        decay = decay * mask.view(1, N, N, 1)

        h = torch.einsum('bijn,bjd->bin', decay, Bu)             # (B,N,d_state)
        y = self.C(h) + self.D * x
        gate = torch.sigmoid(self.gate[0](x))
        return self.dropout(self.norm(gate * y + (1 - gate) * x))
```

**Call it with `dt` computed above** for each Mamba block.

---

## 3) Band‑limit or resolution‑condition your Fourier features

Random Fourier features with a high scale (e.g., 10.0) inject frequencies above the 32×32 Nyquist (~16 cycles per axis), which turns into speckle at native resolution.

### Simple fix

Lower the frequency range and count for 32×32:

```python
self.fourier = FourierFeatures(coord_dim=2, num_freqs=64, scale=4.0)
```

### Better: resolution‑aware features

```python
class ResolutionAwareFourier(nn.Module):
    def __init__(self, coord_dim=2, num_freqs=128, max_cycles=16):
        super().__init__()
        f = torch.linspace(1., max_cycles, num_freqs)
        self.register_buffer('freqs', f)
    def forward(self, coords, size_hint=None):
        cutoff = (size_hint // 2) if size_hint is not None else self.freqs.max().item()
        mask   = self.freqs <= cutoff
        f      = self.freqs[mask]                         # (F_eff,)
        x = 2*math.pi*coords[..., :1] * f
        y = 2*math.pi*coords[..., 1:] * f
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)
```

Pass `size_hint=32` during training on 32×32 queries, and `64/96` at inference.

---

## 4) Local cross‑attention + interpolation prior (residual learning)

Sparse interpolation is **local**. Restrict query attention to K nearest inputs and feed **relative offsets**. Optionally compute an **RBF interpolation prior** and predict a residual on top.

```python
def knn_indices(q, x, k=32):
    # q: (B, Nq, 2), x: (B, Nx, 2)
    d2 = torch.cdist(q, x)                                 # (B,Nq,Nx)
    return torch.topk(d2, k, largest=False).indices        # (B,Nq,k)
```

**Inside your forward pass:**

```python
idx = knn_indices(query_coords, input_coords, k=32)        # (B,Nq,k)
# gather neighbor token features and values
nbr_feat  = input_seq.gather(1, idx.unsqueeze(-1).expand(-1,-1,-1,input_seq.size(-1)))
nbr_coord = input_coords.gather(1, idx.unsqueeze(-1).expand(-1,-1,-1,2))
rel = query_coords.unsqueeze(2) - nbr_coord                # (B,Nq,k,2)

# RBF interpolation prior
sigma = 0.1
w = torch.softmax(-(rel**2).sum(-1)/(2*sigma*sigma), dim=-1)  # (B,Nq,k)
prior = (w.unsqueeze(-1) * input_values.gather(1, idx.unsqueeze(-1).expand(-1,-1,-1,3))).sum(2)  # (B,Nq,3)

# small MLP on [query_token || pooled_nbr || rel_stats] to predict residual
residual = self.decoder(query_seq)                           # (B,Nq,3)
rgb = torch.clamp(prior + residual, -1, 1)                   # if training in [-1,1]
```

This “**interpolate + residual**” trick gives the network a strong bias to reproduce colors/low‑freqs and use capacity on edges/textures.

---

## 5) Training hygiene that matters for FM

- **Range & noise scale**: train RGB in `[-1,1]`. Sample `x₀ ~ N(0,1)` so the target velocity scale matches data.
- **EMA**: maintain an EMA of weights and use it for sampling—often a large PSNR boost.
- **Curriculum on query density**: randomize `N_out` between 10–40% and include *dense* queries (~100%) periodically; prevents over‑fitting to 20% sparsity.
- **Dropout**: set dropout to 0 in the SSM; keep modest dropout only in the MLP/decoder.
- **LR & warmup**: AdamW, `lr=2e-4`, `wd=0.05`, cosine decay, 5k warmup steps. Keep grad‑clip at 1.0.
- **Ablations**: (i) center‑grid only, (ii) center‑grid+Morton, (iii) +geom‑dt, (iv) +bandlimit, (v) +KNN. Track PSNR/SSIM at 32×32.

Example FM training core (with `[-1,1]` range):

```python
# target scaling
x1 = output_values * 2.0 - 1.0          # [-1,1]
x0 = torch.randn_like(x1)               # N(0,1)
t  = torch.rand(B, device=device)
xt = (1 - t).view(B,1,1)*x0 + t.view(B,1,1)*x1
u  = x1 - x0

v  = model(xt, output_coords, t, input_coords, input_values*2-1)
loss = F.mse_loss(v, u)
```

---

## 6) Multi‑scale inference checklist

- Build grids with **pixel centers** (`make_center_grid`).
- Pass the **size hint** to your Fourier features (if using the resolution‑aware version).
- The SSM should receive **geometry‑based dt** even when the query grid is dense.
- Keep the EMA weights for sampling; use Heun or RK with ~50–100 steps.

---

## 7) Debugging checklist (quick wins)

- ✅ 32×32 looks noisy but 64/96 look OK → misaligned grid and/or frequency aliasing.  
- ✅ 32×32 improves when you reduce Fourier scale → too much high‑freq in PE.  
- ✅ Changing the number/order of tokens changes quality → SSM depends on `N`; make it geometric.  
- ✅ Colors off at sparse points → add RBF prior + residual; make attention local and use relative positions.

---

## Appendix A – Minimal code patches

### A.1 Center grid

```python
full_coords = make_center_grid(32, device)
multi_scale_grids = {s: make_center_grid(s, device) for s in [32,64,96]}
```

### A.2 Forward ordering + per‑step `dt` (sketch)

```python
all_coords = torch.cat([input_coords, query_coords], 1)
perm = morton_order(all_coords); inv = torch.argsort(perm, 1)

seq = torch.cat([input_tokens, query_tokens], 1)
seq = torch.gather(seq, 1, perm.unsqueeze(-1).expand(-1,-1,seq.size(-1)))

# per-step dt
sorted_c = torch.gather(all_coords, 1, perm.unsqueeze(-1).expand(-1,-1,2))
diffs = sorted_c[:,1:]-sorted_c[:,:-1]
ds = torch.norm(diffs, -1)
dt = torch.nn.functional.pad(ds, (1,0), value=0.0)
dt = dt / (dt.mean(dim=1, keepdim=True)+1e-8)

for blk in self.mamba_blocks:
    seq = blk.ssm(seq, dt=dt)   # expose dt in your SSM block
    seq = blk.post(seq)

seq = torch.gather(seq, 1, inv.unsqueeze(-1).expand(-1,-1,seq.size(-1)))
```

### A.3 RBF prior + residual (decoder output)

```python
# prior computed from KNN neighbors (as above)
rgb = torch.clamp(prior + self.decoder(query_seq), -1, 1)
```

---

### Why high‑res looked better before these fixes

- With `dt = 1/N`, more tokens → smaller steps → the SSM behaves like a smoother low‑pass at 64/96, masking noise that is obvious at 32×32.
- Fourier features with frequencies beyond the 32×32 Nyquist alias into speckle; oversampling visually “anti‑aliases” them.

**After the changes**, the model converges at 32×32 while preserving the nice continuous‑field super‑resolution capacity.

---

*Prepared guide – drop-in markdown for your repo/docs.*
