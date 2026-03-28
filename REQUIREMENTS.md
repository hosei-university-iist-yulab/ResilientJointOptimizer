# Implementation Requirements Checklist

## Status Legend
- ✔Ready
- ⚠️ Needs clarification
- ╳ Missing / Blocked

---

## 1. Data Requirements

### 1.1 Power System Datasets

| Dataset | Status | Source | Action Needed |
|---------|--------|--------|---------------|
| IEEE 14-bus | ✔Ready | MATPOWER / PyPSA | Built into libraries |
| IEEE 39-bus | ✔Ready | MATPOWER / PyPSA | Built into libraries |
| IEEE 118-bus | ✔Ready | MATPOWER / PyPSA | Built into libraries |
| IEEE 2000-bus | ⚠️ Check | Synthetic / TAMU | Verify availability |

### 1.2 Communication Datasets

| Dataset | Status | Action Needed |
|---------|--------|---------------|
| GridLAB-D co-simulation | ⚠️ Setup required | Install GridLAB-D, create co-sim scenarios |
| Synthetic delays | ✔Ready | Generate from distributions |
| CAIDA traces | ⚠️ Optional | Apply for academic license |
| Mininet-WiFi | ⚠️ Optional | Install if needed for ablation |

### 1.3 Data Generation Scripts Needed

```
□ scripts/generate_synthetic_delays.py
□ scripts/load_ieee_cases.py
□ scripts/gridlabd_interface.py (if using GridLAB-D)
```

---

## 2. Software Dependencies

### 2.1 Core ML Stack

| Package | Version | Status | Purpose |
|---------|---------|--------|---------|
| Python | ≥3.9 | ✔| Runtime |
| PyTorch | ≥2.0 | ✔| Deep learning |
| PyTorch Geometric | ≥2.3 | ✔| GNN layers |
| NumPy | ≥1.24 | ✔| Numerical |
| SciPy | ≥1.10 | ✔| Eigenvalue computation |

### 2.2 Power System Libraries

| Package | Version | Status | Purpose |
|---------|---------|--------|---------|
| pandapower | ≥2.13 | ✔| Power flow, IEEE cases |
| PyPSA | ≥0.25 | ⚠️ Alternative | Power system analysis |
| MATPOWER (via Oct2Py) | optional | ⚠️ | MATLAB interface |

### 2.3 Optimization

| Package | Version | Status | Purpose |
|---------|---------|--------|---------|
| cvxpy | ≥1.4 | ⚠️ Check | Convex optimization |
| Gurobi | optional | ⚠️ License | OPF solver (academic free) |
| OSQP | ≥0.6 | ✔| QP solver (open source) |

### 2.4 Utilities

| Package | Version | Status | Purpose |
|---------|---------|--------|---------|
| wandb | ≥0.15 | ✔| Experiment tracking |
| matplotlib | ≥3.7 | ✔| Visualization |
| networkx | ≥3.1 | ✔| Graph utilities |
| pyyaml | ≥6.0 | ✔| Config files |
| tqdm | ≥4.65 | ✔| Progress bars |

### 2.5 requirements.txt (Draft)

```
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0
scipy>=1.10.0
pandapower>=2.13.0
networkx>=3.1.0
matplotlib>=3.7.0
pyyaml>=6.0.0
wandb>=0.15.0
tqdm>=4.65.0
cvxpy>=1.4.0
osqp>=0.6.0
```

---

## 3. Hardware Requirements

| Component | Minimum | Recommended | Status |
|-----------|---------|-------------|--------|
| GPU | RTX 3060 (12GB) | RTX 3090 (24GB) | ⚠️ Verify available |
| CPU | 8 cores | 16+ cores | ✔|
| RAM | 32 GB | 64 GB | ⚠️ Verify |
| Storage | 50 GB | 100 GB SSD | ✔|

---

## 4. Theoretical Specifications - Gaps Identified

### 4.1 Constants and Hyperparameters ⚠️ NEED VALUES

| Parameter | Description | Proposed Value | Justification |
|-----------|-------------|----------------|---------------|
| K_i | Coupling constant (Theorem 1) | **❓ TBD** | Derive from control gains |
| τ_max | Delay margin | **❓ TBD per system** | Compute from ω_c |
| α | Stability weight in L_coupling | 1.0 | Start balanced |
| β | Control deviation weight | 0.1 | Start conservative |
| γ | Physics mask strength | 1.0 | Start neutral |
| d | Embedding dimension | 128 | Standard |
| L | GNN layers | 3 | Standard |
| n_heads | Attention heads | 8 | Standard |
| lr | Learning rate | 1e-4 | Standard |
| batch_size | Batch size | 32 | Memory dependent |

### 4.2 Missing Derivations ⚠️

| Item | Status | Action |
|------|--------|--------|
| K_i computation from system matrices | ╳ Missing | Derive K_i = ||B_i · K_controller|| |
| τ_max computation per generator | ╳ Missing | Implement from transfer function |
| λ_min(0) computation | ⚠️ Standard | Use scipy.linalg.eig on A matrix |
| Entropy rate H(P_e) estimation | ╳ Missing | Need estimation method |

### 4.3 Architecture Details ⚠️ NEED SPECIFICATION

```
Questions to resolve:
□ How many GNN layers before attention?
□ How many attention layers?
□ Separate encoders for energy and communication, or shared?
□ How to combine GNN and attention outputs?
□ Decoder architecture for control output?
```

**Proposed Architecture (NEED CONFIRMATION):**

```
Input Features:
├── Energy: [P_i, Q_i, V_i, θ_i, ω_i] per bus (5 features)
├── Communication: [τ_ij, R_ij, B_ij] per link (3 features)
└── Control: [u_i^prev] per controllable bus (1 feature)

Encoder:
├── Energy Encoder: Linear(5, d) → GNN(3 layers) → h_E ∈ ℝ^{n×d}
├── Comm Encoder: Linear(3, d) → GNN(3 layers) → h_I ∈ ℝ^{m×d}
└── Cross-Attention: h_E attends to h_I with M_physics mask

Fusion:
├── h_joint = Concat([h_E, CrossAttn(h_E, h_I)])
└── Causal Attention with M_causal mask

Decoder:
├── u_pred = MLP(h_joint) → control actions
└── τ_pred = MLP(h_joint) → predicted delays (optional)

Output: (u, τ_pred)
```

---

## 5. Implementation Roadmap

### Phase 1: Infrastructure (Week 1)

```
□ Set up project structure
□ Create requirements.txt and environment
□ Implement data loaders for IEEE cases
□ Implement synthetic delay generator
□ Write config system (YAML)
□ Set up wandb logging
```

### Phase 2: Core Components (Week 2-3)

```
□ Implement physics computations:
  □ Power flow Jacobian
  □ Eigenvalue computation (λ_min)
  □ Delay margin τ_max
  □ Coupling constants K_i

□ Implement losses:
  □ L_E (OPF loss)
  □ L_I (communication loss)
  □ L_coupling (log-barrier + control deviation)
  □ L_align (contrastive)

□ Implement model:
  □ Energy encoder (GNN)
  □ Communication encoder (GNN)
  □ Cross-domain attention with M_physics
  □ Causal attention with M_causal
  □ Decoder
```

### Phase 3: Training Pipeline (Week 4)

```
□ Training loop
□ Validation loop
□ Checkpointing
□ Early stopping
□ Learning rate scheduling
```

### Phase 4: Baselines (Week 5)

```
□ B1: Sequential OPF + QoS
□ B2: MLP Joint
□ B3: GNN-only
□ B4: LSTM Joint
□ B5: CNN Joint
□ B6: Vanilla Transformer
□ B7: Transformer (no L_coupling)
```

### Phase 5: Experiments (Week 6-7)

```
□ Theorem 1 validation
□ Theorem 2 validation
□ Main comparison
□ Ablation study
□ Scalability study
□ Attention visualization
```

### Phase 6: Paper (Week 8)

```
□ Generate figures
□ Write paper
□ Internal review
```

---

## 6. Open Questions ❓

### 6.1 Critical (Must Resolve Before Starting)

| # | Question | Options | Recommendation |
|---|----------|---------|----------------|
| Q1 | How to compute K_i from power system model? | a) From control Jacobian, b) Learn it, c) Set empirically | **a) From Jacobian** |
| Q2 | How to handle variable grid topology? | a) Fixed graph, b) Dynamic graph, c) Batch by topology | **a) Fixed per dataset** |
| Q3 | Real-time or batch optimization? | a) Batch (train offline), b) Online | **a) Batch first** |
| Q4 | Joint or alternating training? | a) End-to-end, b) Alternating | **a) End-to-end** |

### 6.2 Important (Can Decide During Implementation)

| # | Question | Default |
|---|----------|---------|
| Q5 | Contrastive batch construction? | Random negatives within batch |
| Q6 | Warm-start strategy? | Initialize from previous time step |
| Q7 | Graduated ρ_min schedule? | Linear decrease over epochs |

### 6.3 Nice-to-Have (Defer if Time-Constrained)

| # | Feature | Priority |
|---|---------|----------|
| Q8 | GridLAB-D integration | Medium |
| Q9 | Real-time inference | Low |
| Q10 | Multi-GPU training | Low |

---

## 7. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Eigenvalue computation slow | Medium | High | Use Lanczos, cache results |
| Log-barrier causes NaN | High | High | Clip ρ(τ) ≥ ε_min |
| Attention memory blowup | Medium | Medium | Sparse attention for large grids |
| Theorem 1 bound too loose | Medium | Medium | Report empirical constant |
| No performance gain over baselines | Low | Critical | Start with simpler versions |

---

## 8. Pre-Implementation Checklist

### Must Have ✔

- [ ] Python environment set up
- [ ] PyTorch + PyG installed and tested
- [ ] pandapower installed and IEEE cases loading
- [ ] GPU available and tested
- [ ] Q1-Q4 answered (architecture decisions)
- [ ] K_i and τ_max computation methods defined

### Should Have ⚠️

- [ ] wandb project created
- [ ] Config system designed
- [ ] Unit test framework set up

### Nice to Have 📋

- [ ] GridLAB-D installed
- [ ] CAIDA dataset access
- [ ] Gurobi license

---

## 9. Immediate Next Steps

1. **Answer Q1-Q4** (architecture decisions)
2. **Set up Python environment** with dependencies
3. **Test IEEE case loading** with pandapower
4. **Implement K_i and τ_max computation** (validate Theorem 1)
5. **Start with minimal prototype** (single IEEE 14-bus, synthetic delays)

---

## Confirmed Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Hardware** | RTX 3090 (24GB) | Can handle IEEE 2000-bus, full batch sizes |
| **Scope** | Full pipeline | All IEEE cases + GridLAB-D from start |
| **K_i computation** | Learn from data | Treat as learnable parameter per generator |
| **Timeline** | 8 weeks full | Complete implementation + paper ready |

### K_i as Learnable Parameter

```python
class LearnableCouplingConstants(nn.Module):
    def __init__(self, n_generators):
        super().__init__()
        # Initialize K_i near theoretical estimate, allow fine-tuning
        self.log_K = nn.Parameter(torch.zeros(n_generators))  # log for positivity

    def forward(self):
        return torch.exp(self.log_K)  # K_i > 0 always
```

This approach:
- Guarantees K_i > 0 via exp transform
- Allows data-driven fine-tuning
- Can initialize from analytical estimate if available
