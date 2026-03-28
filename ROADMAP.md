# Implementation Roadmap: Multi-Impairment Stability Bound

**Total: 6 phases, ~30 new files, 0 existing files modified**
**STATUS: ALL 6 PHASES IMPLEMENTED. Full experiments running on GPU 3-6.**

All base paper code stays UNTOUCHED. `JointOptimizer` = B10 baseline.

---

## Phase 0: Configuration & Data Foundation
**Goal:** Generate 6-dim communication data `[tau, R, B, p, sigma_j, s]`
**Depends on:** Nothing (standalone)
**Test:** Generate data for all 8 cases, verify shapes and distributions

### Step 0.1: Impairment Generator
- **File:** `src/data/impairment_generator.py` (NEW)
- **Class:** `ImpairmentGenerator`
- **What:** Sample packet loss `p ~ Beta(2, 20)`, jitter `sigma_j ~ Gamma(2, 15)`, clipped to valid ranges
- **Interface:** `generate(n_scenarios, n_generators) -> (p, sigma_j)` tensors
- **Validation:** Mean p ~ 9%, mean sigma_j ~ 30ms, all within bounds

### Step 0.2: Markov Channel Simulator
- **File:** `src/data/channel_simulator.py` (NEW)
- **Class:** `ChannelSimulator`
- **What:** 3-state Markov chain (GOOD/DEGRADED/FAILED) with configurable transition matrix
- **Interface:** `simulate(n_scenarios, n_generators) -> state` tensor (0/1/2)
- **Validation:** Stationary distribution matches theory, state transitions are realistic

### Step 0.3: Resilient Dataset
- **File:** `src/data/resilient_dataset.py` (NEW)
- **Class:** `ResilientPowerGridDataset`
- **What:** Extends `PowerGridDataset` pattern but outputs 6-dim `comm_x` and adds `packet_loss`, `jitter`, `channel_state` to each sample
- **Uses:** `IEEECaseLoader` (unchanged), `SyntheticDelayGenerator` (unchanged), `ImpairmentGenerator` (new), `ChannelSimulator` (new)
- **Validation:** `dataset[0]['comm_x'].shape == [n_bus, 6]`, all 8 cases load correctly

### Step 0.4: Resilient Config
- **File:** `configs/resilient.yaml` (NEW)
- **What:** Model config with `comm_input_dim: 6`, impairment distribution params, channel model settings, R/J init scales
- **Validation:** Can be loaded and parsed

**Phase 0 deliverable:** `ResilientPowerGridDataset` generates correct 6-dim data for all 8 real cases (39-2869 buses).

---

## Phase 1: Core Model — ResilientJointOptimizer
**Goal:** Build the new model that computes `rho(tau, p, sigma_j)`
**Depends on:** Phase 0 (needs 6-dim data to test)
**Test:** Forward pass produces correct output shapes, rho matches hand-computed values

### Step 1.1: Multi-Impairment Coupling Constants
- **File:** `src/models/multi_impairment_coupling.py` (NEW)
- **Class:** `MultiImpairmentCoupling`
- **What:** Three families of learnable positive constants: `log_K`, `log_R`, `log_J` (all via exp() for positivity)
- **Interface:** `forward() -> (K, R, J)` all `[n_generators]`
- **Init:** Unequal budget split: K gets 50%, R gets 30%, J gets 20% of `s * |lambda_min| / n_g`
- **Validation:** All outputs positive, sum preserves stability budget

### Step 1.2: Channel State Encoder
- **File:** `src/models/channel_model.py` (NEW)
- **Class:** `ChannelStateEncoder`
- **What:** MLP that maps one-hot Markov state `[3]` -> embedding `[embed_dim]`
- **Interface:** `forward(state_indices) -> embeddings`
- **Validation:** Output shape correct, differentiable

### Step 1.3: Resilient Joint Optimizer
- **File:** `src/models/resilient_optimizer.py` (NEW)
- **Class:** `ResilientJointOptimizer`
- **What:** The main model. Uses:
  - `DualDomainGNN` from `gnn.py` (UNTOUCHED) with `comm_input_dim=6`
  - `HierarchicalAttention` from `attention.py` (UNTOUCHED)
  - `ControlDecoder` from `joint_optimizer.py` (UNTOUCHED, imported)
  - `MultiImpairmentCoupling` (Step 1.1)
  - `ChannelStateEncoder` (Step 1.2)
- **Forward signature:** Same as `JointOptimizer` + `p`, `sigma_j`, `sigma_max`
- **Rho computation:**
  ```
  rho = |lambda_min_0| - (K * tau/tau_max).sum(-1)
                        - (R * p/(1-p)).sum(-1)
                        - (J * sigma_j**2/sigma_max**2).sum(-1)
  ```
- **Outputs:** All of `JointOptimizer`'s outputs + `R`, `J`, `channel_pred`
- **Validation:**
  - With `p=0, sigma_j=0`: output matches base `JointOptimizer` exactly
  - Hand-computed rho for known K, R, J, tau, p, sigma_j matches output
  - Gradient flows through all three constant families

### Step 1.4: Update `src/models/__init__.py`
- Export `ResilientJointOptimizer`, `MultiImpairmentCoupling`, `ChannelStateEncoder`
- Keep ALL existing exports unchanged

**Phase 1 deliverable:** `ResilientJointOptimizer` forward pass works, gradients flow, rho is correct.

---

## Phase 2: Loss & Training
**Goal:** Train the model end-to-end
**Depends on:** Phase 0 + Phase 1
**Test:** Training loop converges, K/R/J all learn, stability rate improves

### Step 2.1: Channel State Loss
- **File:** `src/losses/channel_loss.py` (NEW)
- **Class:** `ChannelStateLoss`
- **What:** Cross-entropy loss for Markov state prediction
- **Interface:** `forward(predicted_logits, true_state) -> loss`
- **Validation:** Loss decreases when predictions improve

### Step 2.2: Resilient Joint Loss
- **File:** `src/losses/resilient_loss.py` (NEW)
- **Class:** `ResilientJointLoss`
- **What:** Combines existing losses + `ChannelStateLoss`. Reuses `EnergyLoss`, `CommunicationLoss`, `CouplingLoss`, `PhysicsAwareContrastiveLoss` (all UNTOUCHED) by instantiating them internally.
- **Key:** The `CouplingLoss` receives `rho` from the model — since the model now computes multi-impairment rho, the log-barrier automatically enforces the extended bound. No modification needed.
- **Validation:** All loss components are finite, total loss decreases during training

### Step 2.3: Training Script
- **File:** `scripts/train_resilient.py` (NEW)
- **What:** Training loop for `ResilientJointOptimizer` using `ResilientPowerGridDataset` and `ResilientJointLoss`
- **Args:** `--case`, `--seed`, `--epochs`, `--config configs/resilient.yaml`
- **Logs:** Per-epoch: L_total, rho_mean, stability_rate, K_mean, R_mean, J_mean
- **Validation:** Train on IEEE 39 for 50 epochs, verify stability_rate > 0, K/R/J all change from init

### Step 2.4: K/R/J Diagnostics
- **File:** `src/utils/krj_diagnostics.py` (NEW)
- **What:** Track K_i, R_i, J_i per generator over training. Compute convergence, ratios, dominance.
- **Validation:** Values logged correctly, no NaN/Inf

**Phase 2 deliverable:** Model trains end-to-end on all 8 cases. K, R, J all converge. Stability rate > 95%.

---

## Phase 3: New Baselines (B10-B12, B-ROPF/SOPF/Hinf)
**Goal:** All 16 models runnable on the same dataset
**Depends on:** Phase 0 (needs 6-dim data)
**Test:** All 16 models produce rho values on all 8 cases

### Step 3.1: B10 — Delay-Only Wrapper
- **File:** `src/baselines/delay_only_joint.py` (NEW)
- **Class:** `DelayOnlyJointOptimizer`
- **What:** Wraps existing `JointOptimizer` to accept 6-dim `comm_x` but only uses first 3 dims `[tau, R, B]`. Computes `rho(tau)` only — ignores p, sigma_j.
- **Why wrapper:** The base `JointOptimizer` expects `comm_input_dim=3`. This wrapper extracts the first 3 features from 6-dim data, passes to the original model, and returns results in the same format as `ResilientJointOptimizer`.
- **Validation:** Output matches base `JointOptimizer` exactly when given same data

### Step 3.2: B11 — Naive Multi-Impairment
- **File:** `src/baselines/naive_multi_impairment.py` (NEW)
- **Class:** `NaiveMultiImpairment`
- **What:** Same architecture as `ResilientJointOptimizer` but K, R, J are FIXED (not learned). Values set from theoretical formulas: `K = cond(V) * ||dJ/dtau||`, etc.
- **Validation:** K/R/J don't change during training (frozen parameters)

### Step 3.3: B12 — TCP Retransmit
- **File:** `src/baselines/tcp_retransmit.py` (NEW)
- **Class:** `TCPRetransmitModel`
- **What:** Models packet loss as additional delay: `tau_eff = tau + n_retransmit * RTT`. Uses base `JointOptimizer` with modified delay input.
- **Validation:** `tau_eff > tau` when `p > 0`

### Step 3.4: B-ROPF — Robust OPF
- **File:** `src/baselines/robust_opf.py` (NEW)
- **Class:** `RobustOPF`
- **What:** Worst-case optimization: `min_u max_{tau,p,sigma_j in uncertainty_set} L_OPF(u)`
- **Validation:** Produces feasible dispatch under worst-case

### Step 3.5: B-SOPF — Stochastic OPF
- **File:** `src/baselines/stochastic_opf.py` (NEW)
- **Class:** `StochasticOPF`
- **What:** Monte Carlo scenario-based OPF: sample N impairment scenarios, optimize expected cost
- **Validation:** Cost decreases with more scenarios

### Step 3.6: B-Hinf — H-infinity Controller
- **File:** `src/baselines/hinf_controller.py` (NEW)
- **Class:** `HInfController`
- **What:** LMI-based robust controller with structured uncertainty for delay/loss/jitter
- **Validation:** Satisfies H-infinity norm bound

### Step 3.7: Update `src/baselines/__init__.py`
- Register all new baselines (B10-B12, B-ROPF/SOPF/Hinf)
- Keep ALL existing B1-B9 exports unchanged

**Phase 3 deliverable:** All 16 models can be instantiated and run forward pass on same data.

---

## Phase 4: Experiments
**Goal:** Run all 8 experiments, generate paper tables and figures
**Depends on:** Phase 2 + Phase 3
**Test:** All results files generated, statistical tests pass

### Step 4.1: 16-Model Comparison (E1)
- **File:** `experiments/resilient_baseline_comparison.py` (NEW)
- **What:** Train all 16 models on all 8 cases × 20 seeds. Compute rho mean±std, stability rate. Wilcoxon pairwise tests. Friedman ranking.
- **Output:** `results/resilient_baselines/` — JSON per case per model per seed
- **Table 1:** 16 models × 8 cases

### Step 4.2: Ablation Ki -> Ki+Ri -> Ki+Ri+Ji (E2)
- **File:** `experiments/run_resilient_ablation.py` (NEW)
- **What:** Train 3 variants: (a) only K learned, R=J=0 fixed; (b) K+R learned, J=0; (c) K+R+J all learned. Report safety overestimation % at each stage.
- **Output:** `results/resilient_ablation/`
- **Table 2:** Ablation results

### Step 4.3: Theorem 2 Validation (E3)
- **File:** `experiments/validate_theorem2.py` (NEW)
- **What:** Independent DDE simulation with packet loss (sample-and-hold) and jitter. Compare `rho_DDE` vs `rho_theorem2`. Sweep delay/loss/jitter ranges.
- **Cases:** 39, 118, 1354, 2869
- **Output:** `results/theorem2_validation/`
- **Figure 1:** rho_theoretical vs rho_DDE

### Step 4.4: Safety Overestimation (E4)
- **File:** `experiments/safety_overestimation.py` (NEW)
- **What:** Compare B10 (`rho_delay_only`) vs Ours (`rho_multi`). Compute overestimation %, false safety rate `P(rho_B10 > 0 AND rho_Ours < 0)`.
- **Output:** `results/overestimation/`
- **Table in paper:** Headline numbers

### Step 4.5: Scalability (E5)
- **File:** `experiments/resilient_inference_benchmark.py` (NEW)
- **What:** Measure training time, inference latency (mean/p95/p99), memory, param count for Ours vs B10 across all 8 cases.
- **Output:** `results/resilient_scalability/`
- **Table 4:** Overhead ratios

### Step 4.6: Impairment Sweeps (E6)
- **File:** `experiments/impairment_sweep.py` (NEW)
- **What:** Fix delay, sweep `p = 0% -> 50%`; fix p, sweep `sigma_j = 0 -> 200ms`. Plot rho degradation curves.
- **Cases:** 39, 118, 1354
- **Output:** `results/impairment_sweeps/`
- **Figure 2:** Sweep curves

### Step 4.7: Markov Sensitivity (E7)
- **File:** `experiments/channel_state_eval.py` (NEW)
- **What:** (a) Channel state prediction accuracy. (b) Mean rho per state. (c) Sweep transition probs +/-50%.
- **Cases:** 39, 1354
- **Output:** `results/channel_eval/`

### Step 4.8: Cross-Term Analysis (E8)
- **File:** (included in `validate_theorem2.py` as a flag `--correlated`)
- **What:** Generate correlated impairments (tau, p, sigma_j all high simultaneously). Measure `|rho_additive - rho_DDE|`.
- **Output:** `results/crossterm/`

**Phase 4 deliverable:** All results JSON files, ready for paper generation.

---

## Phase 5: Visualization & Paper Tables
**Goal:** Generate all 5 tables and 3 figures for the paper
**Depends on:** Phase 4
**Test:** PDF figures render, LaTeX tables compile

### Step 5.1: Resilient Visualization
- **File:** `src/utils/resilient_visualization.py` (NEW)
- **What:** Plot functions for:
  - Theorem 2 validation (scatter: rho_theoretical vs rho_DDE)
  - Impairment sweep curves (rho vs p, rho vs sigma_j)
  - Per-impairment stacked bar (K-term, R-term, J-term per generator)
  - K/R/J learning curves over epochs
  - Vulnerability heatmap per generator

### Step 5.2: LaTeX Table Generation
- **File:** `scripts/generate_resilient_tables.py` (NEW)
- **What:** Generate Tables 1-5 from result JSONs
- **Output:** `results/tables/*.tex`

### Step 5.3: Paper Figures
- **Output:** `paper/IEEE-Transactions/figures/fig_theorem2_validation.pdf`, etc.

**Phase 5 deliverable:** Publication-ready figures and tables.

---

## Phase 6: Statistical Validation & Polish
**Goal:** Ensure all claims are statistically sound
**Depends on:** Phase 4
**Test:** All p-values < 0.01, all effect sizes reported

### Step 6.1: Statistical Summary
- **File:** `scripts/run_statistical_tests.py` (NEW)
- **What:** Wilcoxon (pairwise Ours vs each baseline), Friedman (16-model ranking), Cohen's d, Holm-Sidak correction. 20 seeds per comparison.
- **Output:** `results/tables/statistical_summary.tex` (Table 5)

### Step 6.2: Final Verification
- Verify B10 output matches base `JointOptimizer` exactly (no regression)
- Verify `p=0, sigma_j=0` gives same rho as B10 (strict generalization)
- Verify all 8 cases × 20 seeds complete without errors

**Phase 6 deliverable:** Paper-ready statistical tables, all claims verified.

---

## Summary: New Files by Phase

| Phase | New Files | Count |
|-------|-----------|-------|
| 0 | impairment_generator, channel_simulator, resilient_dataset, resilient.yaml | 4 |
| 1 | multi_impairment_coupling, channel_model, resilient_optimizer, __init__ update | 4 |
| 2 | channel_loss, resilient_loss, train_resilient, krj_diagnostics | 4 |
| 3 | delay_only_joint, naive_multi_impairment, tcp_retransmit, robust_opf, stochastic_opf, hinf_controller, __init__ update | 7 |
| 4 | resilient_baseline_comparison, run_resilient_ablation, validate_theorem2, safety_overestimation, resilient_inference_benchmark, impairment_sweep, channel_state_eval | 7 |
| 5 | resilient_visualization, generate_resilient_tables | 2 |
| 6 | run_statistical_tests | 1 |
| **Total** | | **~29 new files** |

**Existing files modified: 0**
**Existing files deleted: 1** (synthetic_grid.py — already done)

---

## Execution Order

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 4 ──► Phase 5
                │                       ▲          ▲
                └──► Phase 3 ───────────┘          │
                                                   │
                                        Phase 6 ───┘
```

- Phase 0 and Phase 1 are sequential (1 needs 0's data)
- Phase 2 needs Phase 0 + 1
- Phase 3 needs Phase 0 only (baselines just need 6-dim data)
- Phase 2 and 3 can run in parallel after Phase 1
- Phase 4 needs Phase 2 + 3
- Phase 5 and 6 need Phase 4

**Critical path:** Phase 0 → 1 → 2 → 4 → 5

---

**Last Updated:** March 22, 2026
