# Topic 2: Communication-Resilient Grid Control Under Packet Loss and Jitter

**Paper Type:** Journal Article (Full Paper)
**Current Status:** Implementation complete — validation and paper writing phase
**Relationship:** Extension of Topic 1 (base paper, under review at Applied Energy; [preprint: SSRN 6368058](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6368058))
**Project Duration:** January 19 → March 22, 2026

---

## Publication Target

### Primary Target: **IEEE Transactions on Smart Grid**

| Aspect | Details |
|--------|---------|
| **Journal** | IEEE Transactions on Smart Grid |
| **Publisher** | IEEE |
| **Impact Factor** | 9.8 (2025) — Top-tier smart grid journal |
| **Page Limit** | 15 pages + 3 pages appendix |
| **Review Time** | 8-16 weeks (expected) |
| **Scope Fit** | Excellent for communication-resilient grid stability |

**Why IEEE Trans. Smart Grid:**
- Base paper is under review at **Applied Energy (Elsevier)** ([preprint available on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6368058)) — different publisher, different journal, NO conflict risk
- IEEE Trans. Smart Grid is the natural home for cyber-physical grid resilience research
- Strong track record for communication-aware power system control
- Reviewers are familiar with the delay-stability coupling problem space
- Different publisher than base paper signals this is an independent, complementary contribution

### Alternatives Considered

| Journal | Pros | Cons | Decision |
|---------|------|------|----------|
| **IEEE Trans. Power Systems** | Strong for stability theory | Less focused on communication aspects | Backup option |
| **IEEE Trans. Communications** | Good for Markov channel model | Less visibility in power systems community | If Markov contribution is emphasized |
| **Applied Energy** | High IF (11.2), accepts long papers | Same journal as base paper — potential overlap concern | **Rejected** |

---

## Paper Information

### Title
**"Communication-Resilient Grid Control Under Packet Loss and Jitter: A Multi-Impairment Stability Bound with Learnable Sensitivity Constants"**

### Authors
- **Franck Junior Aboya Messou** (Hosei University) — First Author
- **Jinhua Chen** (Hosei University)
- **Alaa Zain** (Hosei University)
- **Weiyu Wang** (Hosei University)
- **Keping Yu** (Hosei University)
- **Zihan Zhao** (The University of Osaka)
- **Amr Tolba** (King Saud University)
- **Qiaozhi Hua** (Hubei University of Arts and Science) — Corresponding Author

### Affiliations
- Graduate School of Science and Engineering, Hosei University, Tokyo 184-8584, Japan
- Department of Computer Science and Engineering, King Saud University, Riyadh 11437, Saudi Arabia
- School of Computer Science, Hubei University of Arts and Science, Xiangyang 441053, China

### Funding
- King Saud University Ongoing Research Funding Program
- Computational resources provided by Hosei University

---

## Core Contributions

### 1. Theorem 2: Multi-Impairment Stability Bound — CORE CONTRIBUTION

**Statement:** The stability margin `rho(tau, p, sigma_j)` decreases with communication delay, packet loss, AND jitter, each governed by learnable per-generator constants:

```
rho(tau, p, sigma_j) >= |lambda_min(0)| - SUM_i (Ki * tau_i / tau_max,i)
                                         - SUM_i (Ri * p_i / (1 - p_i))
                                         - SUM_i (Ji * sigma_j,i^2 / sigma_max^2)
```

**Three families of learnable constants:**
- `Ki > 0` — delay sensitivity (from base paper, retained)
- `Ri > 0` — packet loss sensitivity (NEW)
- `Ji > 0` — jitter sensitivity (NEW)

**Critical design decisions (addressing known theoretical issues):**

| Issue | Naive Version | Corrected Version | Why |
|-------|--------------|-------------------|-----|
| **Packet loss term** | `Ri * p_i` (linear) | `Ri * p_i / (1 - p_i)` (nonlinear) | Sample-and-hold analysis gives holding time `tau / (1-p)`, so effective delay perturbation scales as `p/(1-p)`, not `p`. Linear approximation only valid for `p << 1`. |
| **Jitter term** | `Ji * sigma_j` (linear) | `Ji * sigma_j^2 / sigma_max^2` (quadratic) | Jitter is zero-mean noise; first-order expected perturbation vanishes (`E[epsilon] = 0`). The stability impact comes from the second-order term `E[epsilon^2] = sigma^2`. Using sigma linearly is a mathematical error. |
| **Validity domain** | `p in [0, 1]` | `p in [0, p_crit)` where `p_crit < 1` | The term `p/(1-p)` diverges as `p -> 1`. We define `p_crit` as the loss rate where `rho = 0`, and the bound is valid only for `p < p_crit`. This is physically correct: 100% packet loss means no control. |

**Significance:**
- First explicit mathematical bound linking delay, packet loss, AND jitter to power system stability simultaneously
- Ki, Ri, Ji are all learned end-to-end via gradient descent (not manually tuned)
- Strict generalization of Theorem 1: setting `p=0, sigma_j=0` recovers the base paper's bound exactly
- Provides multi-impairment stability certificates for real-time grid control

**Derivation Method:**
1. Steps 1-3 from base paper: Linearize swing equations, Pade approximation of delay-differential equation, Bauer-Fike eigenvalue perturbation for delay term
2. Step 4 (NEW): Packet loss as extended holding delay — lost packet forces zero-order hold on stale command. Expected holding time is `tau_i / (1 - p_i)`, giving Jacobian perturbation `Delta_J_loss = B_i * u_i * p_i / (1 - p_i)`. Apply Bauer-Fike to obtain `Ri * p_i / (1 - p_i)`.
3. Step 5 (NEW): Jitter as stochastic perturbation — delay variation `epsilon_i ~ (0, sigma_j_i)` perturbs the system matrix. Since `E[epsilon] = 0`, the first-order perturbation vanishes in expectation. The stability impact comes from the second moment: `E[||Delta_J_jitter||^2] ~ sigma_j^2 * ||d^2J/dtau^2||^2`. Take square root and apply Bauer-Fike to obtain `Ji * sigma_j^2` after normalization.

**Assumptions (extending base paper):**
- (A1)-(A4): Same as base paper (linearization, bounded delays, diagonalizability, Pade)
- (A5) NEW: Packet loss model — lost packets cause zero-order hold (UDP-like unreliable transport, NO retransmission). This is appropriate for time-critical control where retransmitted packets arrive too late.
- (A6) NEW: Jitter model — per-packet delay is `tau_i + epsilon_i`, with `E[epsilon_i] = 0`, `Var[epsilon_i] = sigma_j_i^2`. Jitter is independent across packets (no burst correlation — see Limitations).
- (A7) NEW: Additive impairment independence — delay, loss, and jitter contribute separate perturbation terms with no cross-terms. See "Anticipated Reviewer Concerns" for justification and planned cross-term experiment.

### 2. Markov Channel Model — NEW

Models the communication channel as a 3-state Markov chain (extended Gilbert-Elliott):

```
         0.95            0.85           0.70
    +----------+    +----------+    +----------+
    |          |    |          |    |          |
    v          |    v          |    v          |
 +---------+  |  +-----------+  |  +--------+  |
 |  GOOD   |--+  | DEGRADED  |--+  | FAILED |--+
 | tau~50ms |    | tau~200ms  |    | p=100%  |
 | p~1%     |--->| p~10%      |--->| (no     |
 | sigma~5ms|0.04| sigma~50ms |0.10| comms)  |
 +---------+    +-----------+    +--------+
      ^              |                |
      |    0.05      |     0.20       |
      +--------------+----------------+
```

**Why 3 states (not 2 like classic Gilbert-Elliott):**
- Classic Gilbert-Elliott has GOOD/BAD. But power grid communications have a distinct "degraded but not failed" regime (congestion, partial interference) that is operationally different from total failure.
- 3 states map to SCADA operational modes: normal monitoring (GOOD), fallback to reduced telemetry (DEGRADED), loss of observability (FAILED).
- The DEGRADED state is where the multi-impairment bound matters most — the system is still controllable but with significant impairments.

**Transition matrix sensitivity:** Default probabilities are illustrative. We include a sensitivity analysis (Experiment E6) sweeping transition probabilities to show bound robustness to +/-50% estimation error.

### 3. Resilient JointOptimizer Architecture

**CRITICAL DESIGN PRINCIPLE: PRESERVE, DON'T REPLACE.**

All existing Topic 1 code (9 baselines + JointOptimizer) is kept **untouched** as-is. The base paper's `JointOptimizer` becomes baseline B10 (delay-only). New Topic 2 models are added **alongside** in new files/classes. This ensures:
- Base paper results are exactly reproducible
- Fair comparison: B10 runs the exact same code as the base paper
- No risk of breaking existing experiments

**What stays untouched (base paper code = B10):**

| File | Class | Role in Topic 2 |
|------|-------|-----------------|
| `src/models/joint_optimizer.py` | `JointOptimizer` | **B10 baseline** (delay-only) |
| `src/models/gnn.py` | `EnergyGNN`, `CommunicationGNN(input_dim=3)`, `DualDomainGNN` | Reused by B10 and inherited by new model |
| `src/models/coupling.py` | `LearnableCouplingConstants` (only `log_K`) | Reused by B10 |
| `src/models/attention.py` | `HierarchicalAttention`, `PhysicsMask`, `CausalMask` | Shared by all models |
| `src/losses/coupling_loss.py` | `CouplingLoss`, `LogBarrierStabilityLoss` | Shared (takes `rho` as input) |
| `src/losses/combined.py` | `JointLoss` | Used by B10 |
| `src/data/dataset.py` | `PowerGridDataset` (3-dim comm_x) | Used by B10 |
| `src/baselines/*.py` | B1-B9 baseline models | All retained for comparison |

**What is NEW (added alongside, never modifying existing files):**

| New File | New Class | What It Does |
|----------|-----------|-------------|
| `src/models/resilient_optimizer.py` **(NEW)** | `ResilientJointOptimizer` | Our model: `comm_input_dim=6`, outputs K/R/J, computes `rho(tau, p, sigma_j)`. Inherits shared components from base. |
| `src/models/multi_impairment_coupling.py` **(NEW)** | `MultiImpairmentCoupling` | 3 families: `log_K`, `log_R`, `log_J`. All positive via exp(). |
| `src/models/channel_model.py` **(NEW)** | `ChannelStateEncoder` | MLP: one-hot(s) -> R^d for Markov state |
| `src/losses/resilient_loss.py` **(NEW)** | `ResilientJointLoss` | `JointLoss` + `L_channel` (wraps/extends, doesn't modify original) |
| `src/losses/channel_loss.py` **(NEW)** | `ChannelStateLoss` | Cross-entropy for Markov state prediction |
| `src/data/resilient_dataset.py` **(NEW)** | `ResilientPowerGridDataset` | 6-dim comm_x `[tau, R, B, p, sigma_j, s]` + impairment generation |
| `src/data/impairment_generator.py` **(NEW)** | `ImpairmentGenerator` | Beta packet loss, Gamma jitter, Markov chain |
| `src/data/channel_simulator.py` **(NEW)** | `ChannelSimulator` | 3-state Markov chain sampling |
| `src/baselines/delay_only_joint.py` **(NEW)** | `DelayOnlyJointOptimizer` | Wrapper making base `JointOptimizer` usable as B10 with 6-dim data (ignores p, sigma_j) |
| `src/baselines/naive_multi_impairment.py` **(NEW)** | `NaiveMultiImpairment` | B11: fixed Ki, Ri, Ji from theory |
| `src/baselines/tcp_retransmit.py` **(NEW)** | `TCPRetransmitModel` | B12: models loss as additional delay |
| `src/baselines/robust_opf.py` **(NEW)** | `RobustOPF` | B-ROPF: worst-case optimization |
| `src/baselines/stochastic_opf.py` **(NEW)** | `StochasticOPF` | B-SOPF: Monte Carlo scenario |
| `src/baselines/hinf_controller.py` **(NEW)** | `HInfController` | B-Hinf: H-infinity robust control |

**Complete model lineup (15 + Ours = 16 models):**

| ID | Model | Source | Type |
|----|-------|--------|------|
| B1-B9 | Base paper baselines | Existing `src/baselines/` (untouched) | Architectural |
| B10 | `JointOptimizer` (delay-only rho) | Existing `src/models/joint_optimizer.py` (untouched) | Base paper's best |
| B11 | Naive Multi-Impairment (fixed K,R,J) | New `src/baselines/naive_multi_impairment.py` | Topic 2 ablation |
| B12 | TCP-Retransmit (loss as delay) | New `src/baselines/tcp_retransmit.py` | Topic 2 ablation |
| B-ROPF | Robust OPF | New `src/baselines/robust_opf.py` | Classical method |
| B-SOPF | Stochastic OPF | New `src/baselines/stochastic_opf.py` | Classical method |
| B-Hinf | H-infinity Controller | New `src/baselines/hinf_controller.py` | Classical method |
| **Ours** | **ResilientJointOptimizer** | New `src/models/resilient_optimizer.py` | **This paper** |

### 4. Resilient Loss Function (NEW class, original untouched)

**Base paper's `JointLoss`** ([combined.py:40](src/losses/combined.py#L40)) is kept as-is for B10.

**New `ResilientJointLoss`** ([resilient_loss.py](src/losses/resilient_loss.py) NEW) wraps and extends it:

```
L_total = L_E + L_I + L_coupling + L_contrastive + L_channel

Where:
  L_E         = EnergyLoss (energy_loss.py)                    — reused, unchanged
  L_I         = CommunicationLoss (communication_loss.py)      — reused, unchanged
  L_coupling  = CouplingLoss (coupling_loss.py)                — reused: receives rho from model
  L_contrastive = PhysicsAwareContrastiveLoss (contrastive.py) — reused, unchanged
  L_channel   = ChannelStateLoss (channel_loss.py)             — NEW: Markov state prediction CE
```

The key insight: `CouplingLoss` takes `rho` as input and applies the log-barrier. Since `ResilientJointOptimizer` computes `rho(tau, p, sigma_j)` instead of `rho(tau)`, the same coupling loss automatically enforces the multi-impairment bound. No modification needed — the change is upstream in how `rho` is computed by the model.

The only addition is `L_channel` for Markov state prediction.

---

## Anticipated Reviewer Concerns and Preemptive Responses

This section addresses the most likely reviewer objections, with planned experiments or arguments for each.

### R1: "The additive bound ignores interaction between impairments"

**Concern:** Delay, packet loss, and jitter are not independent. Lost packets cause stale commands (delay-loss interaction). Jitter correlates with delay (both worsen under congestion).

**Response strategy:**
1. **Theoretical:** The additive bound is a first-order approximation. We prove it is a valid upper bound under assumption (A7) because each perturbation term is derived independently via Bauer-Fike, and the triangle inequality guarantees `||Delta_total|| <= ||Delta_delay|| + ||Delta_loss|| + ||Delta_jitter||`. Cross-terms would tighten the bound but cannot make it invalid (it remains conservative).
2. **Experimental:** We include a dedicated cross-term experiment: compare additive bound vs DDE ground truth across correlated impairment scenarios (congestion: high tau + high p + high sigma simultaneously). If the gap exceeds 15%, we add an optional cross-term `C_ij * tau_i * p_i` as a Remark/Observation (not part of the theorem).
3. **Practical argument:** A conservative (loose) bound is safer than a tight bound that fails under correlation. Grid operators prefer "the system might be more stable than we think" over "we thought it was stable but it wasn't."

### R2: "Novelty is incremental — just adding two more terms to Theorem 1"

**Concern:** The architecture changes are minimal (wider input, two more parameter families). Is this a full paper?

**Response strategy:**
1. **The theory IS the novelty:** The packet loss term requires sample-and-hold analysis (control theory, not trivial). The jitter term requires stochastic perturbation theory (second-order). These are not copy-paste of the delay proof.
2. **The practical impact is large:** We show delay-only overestimates safety by 20-40% under realistic conditions. This is a safety-critical gap, not an incremental improvement.
3. **The Markov channel model** adds a fundamentally new capability (dynamic degradation prediction) absent from the base paper.
4. **Comparison with classical robust control** (new baselines, see below) shows the learning-based approach outperforms H-infinity and stochastic OPF methods that also handle uncertainty.

### R3: "Why not compare with robust/stochastic optimal power flow?"

**Concern:** Classical methods (H-infinity control, stochastic OPF, chance-constrained OPF) handle uncertainty. What does learning add?

**Response strategy:** Add **three external baselines** beyond the architectural baselines:

| Baseline | Method | What It Tests |
|----------|--------|---------------|
| B-ROPF | Robust OPF with worst-case delay/loss margins | ML vs. conservative robust optimization |
| B-SOPF | Stochastic OPF with Monte Carlo sampling | ML vs. sampling-based uncertainty quantification |
| B-Hinf | H-infinity controller with delay/loss uncertainty set | ML vs. classical robust control |

**Expected advantage:** Classical methods use fixed worst-case or distributional assumptions. Our learnable constants adapt to the specific grid topology and operating point, giving tighter bounds without sacrificing safety.

### R4: "Statistical validation is weak with only 5 seeds"

**Concern:** Wilcoxon signed-rank with n=5 has very low statistical power.

**Response:** Use **20 seeds** (same fix as the base paper's reviewer feedback). Seeds: `{0, 42, 84, 126, 168, 210, 252, 294, 336, 378, 420, 462, 504, 546, 588, 630, 672, 714, 756, 798}`. Wilcoxon with n=20 gives minimum achievable p ~ 9.5e-7, well below 0.01 threshold.

### R5: "The Markov transition probabilities are arbitrary"

**Concern:** No empirical basis for the 3x3 transition matrix.

**Response strategy:**
1. Include sensitivity analysis sweeping each transition probability by +/-50% (Experiment E6)
2. Cite empirical SCADA measurement studies for typical link reliability statistics
3. Show that the bound's ranking (which generators are most vulnerable) is robust to transition matrix perturbation even if absolute rho values shift

### R6: "Only tested on small grids / no realistic scale"

**Concern:** Experiments limited to small IEEE cases, not real-world scale.

**Response strategy:**
1. We test on **8 real-world power system models spanning 39 to 2,869 buses** — all loaded from pandapower. The three largest are:
   - PEGASE 1354 (1,354 buses, European grid)
   - RTE 1888 (1,888 buses, French national transmission)
   - PEGASE 2869 (2,869 buses, continental European grid)
2. PEGASE/RTE cases are derived from actual European transmission system data (PEGASE project funded by EU FP7, French RTE operator data). Reviewers cannot dismiss these as toy problems.
3. The ~74x scale range (39 → 2,869 buses) demonstrates robust scalability from regional to continental level.
4. IEEE 39-300 provide standardized small/medium benchmarks that the community expects.
5. Acknowledge real SCADA communication measurement data (ORNL testbed, ICS-CERT) as future validation for the communication impairment models specifically.

### R7: "Same authors, closely related to a paper under review — salami slicing?"

**Concern:** Editor may see this as splitting one paper into two.

**Response strategy:**
1. Base paper is at **Applied Energy (Elsevier)** while this paper targets **IEEE Trans. Smart Grid** — different publisher, different journal, different community emphasis
2. The paper's Introduction clearly delineates: base paper = delay-only bound + architecture; this paper = multi-impairment bound (NEW theorem) + Markov channel model (NEW) + cyber-resilience analysis (NEW application)
3. The theoretical contribution (Steps 4-5 of the proof) is entirely new, not a subset of the base paper
4. Cover letter explicitly states the relationship and explains why this is an independent contribution

### R8: "UDP assumption is unrealistic — real SCADA uses TCP/DNP3"

**Concern:** If packets are retransmitted (TCP), then "packet loss" becomes "additional delay", not "lost information."

**Response strategy:**
1. Explicitly state assumption (A5): the model targets UDP-like unreliable transport, appropriate for time-critical control where retransmitted packets arrive too late to be useful
2. Baseline B12 (TCP-Retransmit Model) explicitly tests this: it models loss as additional delay `tau_eff = tau + n_retransmit * RTT`. We compare: is treating loss as delay sufficient, or does the separate Ri term add value?
3. Note that IEC 61850 GOOSE messages (used for protection/control in modern substations) use multicast UDP, not TCP. This is the primary use case.

---

## Relationship to Base Paper

| Aspect | Base Paper (Topic 1) | This Paper (Topic 2) |
|--------|---------------------|---------------------|
| **Core theorem** | Theorem 1: `rho(tau)` (delay-only) | Theorem 2: `rho(tau, p, sigma_j)` (multi-impairment) |
| **Learnable params** | Ki only (n_g params) | Ki + Ri + Ji (3 * n_g params) |
| **Communication model** | Static delay distribution | Dynamic Markov channel (3 states) |
| **Comm features** | 3-dim `[tau, R, B]` | 6-dim `[tau, R, B, p, sigma_j, s]` |
| **Proof technique** | Pade + Bauer-Fike | Same + sample-and-hold + stochastic perturbation |
| **Transport assumption** | Implicit (delay only) | Explicit: UDP-like (A5), with TCP comparison (B12) |
| **Application** | Stability monitoring | Stability + cyber-resilience assessment |
| **Target journal** | Applied Energy (Elsevier) | IEEE Trans. Smart Grid (different publisher) |
| **Code base** | `topic1-energy-info-cooptimization/` | `topic2-packet-loss-jitter-resilience/` (forked) |

**Generalization:** Setting `p=0, sigma_j=0` in Theorem 2 recovers Theorem 1 exactly — strict generalization, not replacement.

**Preprint risk management:** The base paper is currently under review at Applied Energy, with a [preprint publicly available on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6368058) (Elsevier's preprint server).
- **If accepted as-is:** Ideal — cite the published version.
- **If major revision changes Theorem 1:** Our Theorem 2 proof Steps 1-3 reuse Theorem 1's delay term. If Theorem 1's form changes, we update Steps 1-3 accordingly. Steps 4-5 (our new contributions) are independent.
- **If rejected:** Theorem 1 is still valid mathematics regardless of publication status. We cite the SSRN preprint and include the delay-only bound as a Lemma within this paper's proof.

---

## Experimental Design

### Test Systems (8 Real-World Cases, 39 to 2,869 Buses)

**ALL cases are real power system models** loaded from pandapower via `IEEECaseLoader`.

| Test Case | Buses | Generators | Lines | Origin | Scale |
|-----------|-------|------------|-------|--------|-------|
| IEEE 39 | 39 | 10 | 35 | New England system | Regional grid |
| IEEE 57 | 57 | 7 | 63 | IEEE standard | Mid-size state |
| IEEE 118 | 118 | 54 | 173 | IEEE standard | Metropolitan area |
| IEEE 145 | 145 | 50 | 378 | IEEE standard | Large metropolitan |
| IEEE 300 | 300 | 69 | 283 | IEEE standard | Regional interconnection |
| PEGASE 1354 | 1,354 | 260 | 1,751 | European grid (PEGASE project) | National-scale |
| RTE 1888 | 1,888 | 272 | 1,976 | French transmission (RTE) | National-scale |
| PEGASE 2869 | 2,869 | 510 | 4,051 | European grid (PEGASE project) | Continental-scale |

**Why this selection:**
- IEEE 39-300: **Standard benchmarks** that every reviewer expects. Reproducible, widely compared in the literature.
- PEGASE/RTE 1354-2869: **Real national/continental-scale grids** derived from actual European transmission system data (PEGASE project funded by EU FP7, French RTE operator). Reviewers cannot dismiss these as "toy problems."
- Spanning **39 to 2,869 buses** (~74x range) demonstrates scalability from regional to continental-level grids.
- **All cases are real** — no synthetic grids. All load via the same `IEEECaseLoader` interface from pandapower.

### Data Generation (Extended from Base Paper)

**Energy side** (unchanged):
- Load perturbation: `P_load_j ~ U(0.8, 1.2) * P_base_j`
- Generator redispatch for power balance
- 1000 scenarios per case

**Communication side** (EXTENDED):
- Delay: `tau_i ~ LogNormal(mu=50ms, sigma=20ms)`, clipped to [5, 500ms]
- Packet loss: `p_i ~ Beta(2, 20)` (mean ~9%, right-skewed, clipped to [0, 0.5])
- Jitter: `sigma_j_i ~ Gamma(2, 15)` (mean ~30ms, clipped to [0, 200ms])
- Channel state: sampled from Markov chain with empirical transition matrix

**Distribution justification:**
- Beta for packet loss: bounded on [0,1], right-skewed (most links have low loss). Shape params (2, 20) give mean ~9% matching typical wireless SCADA link degradation.
- Gamma for jitter: non-negative, right-skewed. Shape=2, scale=15 gives mean ~30ms matching measured SCADA network jitter under congestion.
- LogNormal for delay: standard choice in network modeling, matches empirical delay distributions.
- Sensitivity to distribution choice is tested in Experiment E7.

### Baselines (15 Methods)

**Architectural baselines B1-B9 from base paper (retained):**

| Baseline | Architecture | Key Feature |
|----------|-------------|-------------|
| B1: Sequential OPF | Traditional | Energy first, then comm |
| B2: MLP Joint | Fully-connected | No graph structure |
| B3: GNN-only | Message passing | No global attention |
| B4: LSTM Joint | Recurrent | Sequential processing |
| B5: CNN Joint | Convolutional | Fixed receptive field |
| B6: Vanilla Transformer | Standard attention | No physics mask |
| B7: Transformer (no coupling) | Attention + GNN | No L_coupling |
| B8: Transformer (fixed K) | Full architecture | Ki not learned |
| B9: HeteroGNN | Heterogeneous GNN | Different node/edge types |

**New topic2-specific baselines:**

| Baseline | Description | What It Tests |
|----------|------------|---------------|
| B10: Delay-Only JointOptimizer | Base paper's model (ignores p, sigma_j) | How much safety margin is overestimated |
| B11: Naive Multi-Impairment | Fixed Ki, Ri, Ji (not learned, set from theory) | Value of learning vs. analytical constants |
| B12: TCP-Retransmit Model | Models loss as additional delay `tau_eff` | Whether loss can be reduced to delay |

**External method baselines (addressing Reviewer R3):**

| Baseline | Description | What It Tests |
|----------|------------|---------------|
| B-ROPF: Robust OPF | Worst-case optimization over impairment uncertainty set | ML vs. conservative robust optimization |
| B-SOPF: Stochastic OPF | Monte Carlo scenario-based OPF | ML vs. sampling-based approach |
| B-Hinf: H-infinity Controller | Structured singular value with delay/loss uncertainty | ML vs. classical robust control |

### Experiment Types (8 Core Experiments)

Reduced from 12 to 8 to fit realistically within 15 pages + 3-page appendix (~6 experiments in main text, ~2 in appendix).

| # | Experiment | Cases | What It Tests | Pages |
|---|-----------|-------|---------------|-------|
| E1 | Main baseline comparison (16 models) | All 8 | Multi-impairment rho vs. all baselines | Main (0.75p) |
| E2 | Ablation (Ki only -> Ki+Ri -> Ki+Ri+Ji) | All 8 | Contribution of each constant family | Main (0.5p) |
| E3 | Theorem 2 validation via DDE simulation | 39, 118, 1354, 2869 | Non-circular bound validation across scales | Main (0.5p) |
| E4 | Safety overestimation analysis | All 8 | Delay-only (B10) overestimates rho by how much? | Main (0.5p) |
| E5 | Scalability analysis | 39->2869 | Training time, inference time, memory vs grid size | Main (0.25p) |
| E6 | Packet loss + jitter sweep | 39, 118, 1354 | Margin degradation curves | Condensed (0.25p) |
| E7 | Markov transition sensitivity | 39, 1354 | Robustness to +/-50% transition probability error | Condensed (0.25p) |
| E8 | Cross-term analysis | 39, 118 | Does additive approximation hold under correlated impairments? | Supplementary |

### Evaluation Metrics (Streamlined — No Redundancy)

**Removed from base paper** (training plumbing, not results):
- Individual loss components (L_cost, L_voltage, L_frequency, L_latency, L_bandwidth, etc.) — only L_total for training curves
- Contrastive accuracy / positive-negative similarity — debugging, not a result
- Attention entropy / mutual information — fixed gamma issue, no longer needed
- Padé-1 vs Padé-2 error — already validated in base paper
- Gamma sweep — hyperparameter tuning, not a result
- Model compression / transfer learning / economic analysis — distractions from multi-impairment story
- Delay distribution robustness / convergence epochs — base paper's concerns, not ours

**Retained from base paper** (still meaningful):

| Metric | Formula | Role |
|--------|---------|------|
| Stability margin `ρ` | `|λ_min| - K·τ/τ_max - R·p/(1-p) - J·σ²/σ²_max` | Primary outcome metric |
| Stability rate | `% samples with ρ > 0` | Summary statistic |
| Learned constants K_i | `exp(log_K_i)` per generator | Delay sensitivity profile |
| Inference latency | Mean, p95, p99 (ms) | Real-time feasibility |
| Parameter count | Total learnable params | Scalability overhead |
| N-1 contingency | ρ under single line removal | Practical safety |
| Ablation impact | ρ with/without each component | Component contribution |

**NEW for Topic 2** (the metrics that make this paper):

| Metric | Formula | What It Proves |
|--------|---------|---------------|
| **Safety overestimation %** | `(ρ_B10 - ρ_Ours) / ρ_Ours × 100` | Headline: "delay-only overestimates safety by X%" |
| **False safety rate** | `P(ρ_B10 > 0 AND ρ_Ours < 0)` | "X% of scenarios are falsely declared safe" |
| **Bound tightness** | `|ρ_theorem2 - ρ_DDE| / ρ_DDE` | Theorem 2 is valid AND tight |
| **Critical loss threshold `p_crit`** | `p` where `ρ = 0` per generator | "Generator 7 fails at 23% packet loss" |
| **Learned R_i, J_i** | `exp(log_R_i)`, `exp(log_J_i)` per generator | Loss and jitter sensitivity profiles |
| **K:R:J dominance ratio** | Relative magnitude per generator | Which impairment dominates where |
| **Per-impairment contribution** | K-term, R-term, J-term of ρ separately | Stacked bar: breakdown of margin erosion |
| **Vulnerability index** | `V_i = R_i · p_max + J_i · σ²_max` | Ranks generators by attack vulnerability |
| **Channel state prediction** | F1 / accuracy of Markov state prediction | Validates channel model learns dynamics |
| **Margin by channel state** | Mean ρ in GOOD / DEGRADED / FAILED | Quantifies degradation impact |
| **Computational overhead** | `t_Ours / t_B10` for training and inference | Proves minimal cost of extension |

**Paper figures/tables plan (fits 2.5 pages of results):**

| Item | Content | Key Metric |
|------|---------|-----------|
| Table 1 | 16 models × 8 cases: ρ mean±std, stability rate | ρ, stability rate |
| Table 2 | Ablation: Ki → Ki+Ri → Ki+Ri+Ji, per case | Safety overestimation % |
| Table 3 | Per-generator K_i, R_i, J_i, V_i for IEEE 118 and PEGASE 1354 | Vulnerability index |
| Figure 1 | Theorem 2 validation: ρ_theoretical vs ρ_DDE across impairment levels | Bound tightness |
| Figure 2 | Impairment sweeps: ρ vs p (loss) and ρ vs σⱼ (jitter) | Margin degradation |
| Figure 3 | Per-impairment contribution stacked bar across generators | K:R:J breakdown |
| Table 4 | Scalability: time, params, memory from 39 to 2,869 buses | Overhead ratio |
| Table 5 | Statistical significance: Wilcoxon p-values, Cohen's d | p < 0.01, d > 0.8 |

### Statistical Methodology (Strengthened from Base Paper)

- **20 seeds**: `{0, 42, 84, 126, 168, 210, 252, 294, 336, 378, 420, 462, 504, 546, 588, 630, 672, 714, 756, 798}`
- Wilcoxon signed-rank test for pairwise comparisons (p < 0.01 threshold)
- Holm-Sidak correction for multiple comparisons
- Cohen's d effect sizes with 95% confidence intervals
- Friedman test + Nemenyi post-hoc for 16-model ranking

---

---


**Last Updated:** March 28, 2026
