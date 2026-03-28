# Changelog

All notable changes to Topic 2 (Packet Loss / Jitter Resilience) are documented here.

## [0.6.0] - 2026-03-22 — Full Run + Table/Figure Updates

### Running
- Full experiments on GPU 3-6 (concurrent, many processes per GPU)
- 8 cases × 7 models × 20 seeds + E2/E4/E5/E6/E7

### Changed
- `scripts/generate_latex_tables.py`: added 4 resilient tables (R1-R4) with `--resilient` flag
- `scripts/figures/generate_publication_figures.py`: added 4 resilient figures (R1-R4)
- Case labels use `Case-X` / `PEGASE-X` / `RTE-X` instead of `IEEE X` (journal-neutral)
- Existing table/figure functions UNTOUCHED

## [0.5.0] - 2026-03-19 — Phase 5-6: Visualization + Statistics

### Added
- `src/utils/resilient_visualization.py` — 3 new plot functions (existing vis untouched)
- `scripts/run_statistical_tests.py` — Wilcoxon, Cohen's d, Holm-Sidak

## [0.4.0] - 2026-03-16 — Phase 4: Experiment Scripts

### Added
- `experiments/resilient_baseline_comparison.py` — E1: 16-model comparison on all 8 cases
- `experiments/run_resilient_ablation.py` — E2: Ki -> Ki+Ri -> Ki+Ri+Ji ablation
- `experiments/safety_overestimation.py` — E4: B10 vs Ours, false safety rate
- `experiments/impairment_sweep.py` — E6: p and sigma_j sweep curves
- `experiments/channel_state_eval.py` — E7: Markov prediction accuracy, rho per state
- `experiments/resilient_inference_benchmark.py` — E5: scalability benchmark

## [0.3.1] - 2026-03-14 — Critical Fixes

### Fixed
- **Eigenvalue computation**: replaced uniform M=5, D=2 with generator-specific inertia H_i
  derived from machine MW ratings. lambda_min now varies across cases (-0.035 to -0.077).
- **Loss normalization**: applied log1p to prevent energy cost (~4 billion) from dwarfing
  coupling loss (~1.0). Loss now ~22, gradients reach K/R/J.
- **Channel accuracy**: now logged per epoch in training output.

### Changed
- `src/data/ieee_cases.py`: new `get_generator_dynamics()` method with H, D, Ks from pandapower
- `scripts/train_resilient.py`: log1p normalization, channel accuracy in output

## [0.3.0] - 2026-03-11 — Phase 3: Baselines

### Added
- `src/baselines/delay_only_joint.py` — B10: wraps base JointOptimizer for 6-dim data
- `src/baselines/naive_multi_impairment.py` — B11: frozen K/R/J
- `src/baselines/tcp_retransmit.py` — B12: loss as extra delay
- `src/baselines/robust_opf.py` — B-ROPF: worst-case OPF
- `src/baselines/stochastic_opf.py` — B-SOPF: Monte Carlo scenario OPF
- `src/baselines/hinf_controller.py` — B-Hinf: H-infinity controller

## [0.2.0] - 2026-03-08 — Phase 2: Loss & Training

### Added
- `src/losses/channel_loss.py` — ChannelStateLoss (CE for Markov state prediction)
- `src/losses/resilient_loss.py` — ResilientJointLoss (wraps JointLoss + L_channel)
- `scripts/train_resilient.py` — End-to-end training with KRJ logging
- `src/utils/krj_diagnostics.py` — K/R/J convergence tracking

## [0.1.0] - 2026-03-01 — Phase 0+1: Data + Model

### Added
- `src/data/impairment_generator.py` — Beta packet loss, Gamma jitter sampling
- `src/data/channel_simulator.py` — 3-state Markov chain simulator
- `src/data/resilient_dataset.py` — 6-dim comm dataset
- `configs/resilient.yaml` — Multi-impairment configuration
- `src/models/multi_impairment_coupling.py` — K/R/J learnable constants
- `src/models/channel_model.py` — ChannelStateEncoder + Predictor
- `src/models/resilient_optimizer.py` — ResilientJointOptimizer

### Removed
- `src/data/synthetic_grid.py` — deleted (all 8 cases are real)

### Changed
- `src/data/ieee_cases.py`: SUPPORTED_CASES now {39, 57, 118, 145, 300, 1354, 1888, 2869}
- `src/data/dataset.py`: removed synthetic grid routing

## [0.0.0] - 2026-01-19 — Foundation

### Added
- Copied topic1 codebase as starting point
- `README.md` — comprehensive paper plan with reviewer anticipation
- `ROADMAP.md` — 6-phase implementation plan
- `LEARN_THE_PROJECT.md` — pedagogical guide for Idea 2

### Design Decisions
- **Preserve, don't replace**: all base paper code untouched (= B10 baseline)
- **New files alongside**: 0 existing files modified (only __init__.py exports)
- **8 real cases**: IEEE 39-300 + PEGASE/RTE 1354-2869, no synthetic grids
- **16 models**: B1-B9 (base) + B10-B12 + B-ROPF/SOPF/Hinf + Ours
