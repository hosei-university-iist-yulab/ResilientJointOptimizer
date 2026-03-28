#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/13/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Time-Domain Simulation for Independent Theorem 1 Validation

Numerically integrates the delayed swing equation (DDE) using Euler method
with a delay buffer. Extracts the empirical stability margin from the
oscillation envelope decay rate — completely independent of Theorem 1 formula.

This breaks the circular validation where both rho_emp and rho_theo used
the same analytic expression.
"""

import numpy as np
from typing import Tuple, Optional


class DelayedSwingEquationSimulator:
    """
    Simulate dx/dt = A*x(t) + B*x(t-tau) via Euler integration with delay buffer.

    The delayed differential equation (DDE) for the linearized swing equation:
        M*d²δ/dt² + D*dδ/dt = P_m(t-τ) - P_e(t)

    In state-space form:
        dx/dt = A_sys * x(t) + B_delay * x(t - τ)

    where x = [δ₁, ..., δₙ, ω₁, ..., ωₙ]ᵀ
    """

    def __init__(self, dt: float = 0.001, T: float = 10.0):
        """
        Args:
            dt: Integration time step (seconds)
            T: Total simulation time (seconds)
        """
        self.dt = dt
        self.T = T

    def simulate(
        self,
        A_sys: np.ndarray,
        B_delay: np.ndarray,
        tau_vec: np.ndarray,
        x0: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Integrate the DDE using Euler method with per-generator delay buffer.

        Args:
            A_sys: System matrix [2n, 2n]
            B_delay: Delay coupling matrix [2n, 2n]
            tau_vec: Per-generator delays [n_gen] in seconds
            x0: Initial perturbation [2n]. If None, uses small random perturbation.
            seed: Random seed for initial perturbation

        Returns:
            trajectory: State trajectory [n_steps, 2n]
        """
        n_states = A_sys.shape[0]
        n_steps = int(self.T / self.dt)

        # Convert per-generator delays to step counts
        delay_steps = np.maximum((tau_vec / self.dt).astype(int), 1)
        max_delay = int(np.max(delay_steps))

        # Initialize state history buffer (for delayed access)
        # Buffer stores [max_delay + n_steps + 1, n_states]
        total_length = max_delay + n_steps + 1
        x_history = np.zeros((total_length, n_states))

        # Initial perturbation
        if x0 is None:
            rng = np.random.RandomState(seed)
            x0 = rng.randn(n_states) * 0.01
        x_history[max_delay] = x0

        # Euler integration
        for k in range(n_steps):
            idx = k + max_delay
            x_now = x_history[idx]

            # Build delayed state vector:
            # For each generator i, we use x(t - tau_i) for the corresponding
            # state components. The delay matrix B_delay acts on x(t - tau).
            # For simplicity, use the mean delay for the full delayed state.
            # For per-generator delays, construct x_delayed with appropriate offsets.
            n_gen = len(tau_vec)
            x_delayed = np.zeros(n_states)
            for i in range(n_gen):
                d = delay_steps[i]
                # Delta components (first n_gen entries)
                x_delayed[i] = x_history[idx - d, i]
                # Omega components (second n_gen entries)
                if i + n_gen < n_states:
                    x_delayed[i + n_gen] = x_history[idx - d, i + n_gen]

            # Euler step: dx = (A*x(t) + B*x(t-tau)) * dt
            dx = A_sys @ x_now + B_delay @ x_delayed
            x_history[idx + 1] = x_now + dx * self.dt

            # Divergence check
            if np.max(np.abs(x_history[idx + 1])) > 1e6:
                # System unstable — fill remaining with last value
                x_history[idx + 1:] = x_history[idx + 1]
                break

        # Return only the simulation portion (after delay buffer)
        return x_history[max_delay: max_delay + n_steps + 1]

    def extract_decay_rate(
        self,
        trajectory: np.ndarray,
    ) -> float:
        """
        Extract exponential decay rate from oscillation envelope.

        Fits: envelope(t) ~ C * exp(-rho * t)
        Returns rho (positive = stable, negative = unstable).

        Args:
            trajectory: State trajectory [n_steps, n_states]

        Returns:
            rho_empirical: Decay rate (stability margin)
        """
        # Compute envelope: max absolute value across all states at each time step
        envelope = np.max(np.abs(trajectory), axis=1)

        # Time vector
        n_steps = len(envelope)
        t_vec = np.arange(n_steps) * self.dt

        # Filter out near-zero values and initial transient
        # Skip first 5% for transient, use points where envelope > threshold
        start_idx = max(1, n_steps // 20)
        threshold = np.max(envelope) * 1e-8

        valid = envelope[start_idx:] > threshold
        if np.sum(valid) < 10:
            # Too few points — system may have converged to zero (very stable)
            # or diverged. Check final value.
            if envelope[-1] > envelope[0]:
                return -1.0  # Unstable
            else:
                return 10.0  # Very stable (fast decay)

        t_valid = t_vec[start_idx:][valid]
        env_valid = envelope[start_idx:][valid]

        # Linear regression on log(envelope) vs t
        # log(C * exp(-rho*t)) = log(C) - rho*t
        log_env = np.log(env_valid)

        # Use least squares: log_env = a + b*t => rho = -b
        A_mat = np.column_stack([np.ones_like(t_valid), t_valid])
        result = np.linalg.lstsq(A_mat, log_env, rcond=None)
        coeffs = result[0]

        rho_empirical = -coeffs[1]  # Positive means stable
        return float(rho_empirical)


def compute_empirical_margin_independent(
    A_sys: np.ndarray,
    B_delay: np.ndarray,
    tau_vec: np.ndarray,
    dt: float = 0.001,
    T: float = 10.0,
    n_trials: int = 5,
) -> Tuple[float, float]:
    """
    Compute empirical stability margin via time-domain simulation.

    Runs multiple trials with different initial perturbations and averages.

    Args:
        A_sys: System matrix [2n, 2n]
        B_delay: Delay coupling matrix [2n, 2n]
        tau_vec: Per-generator delays [n_gen] in seconds
        dt: Integration time step
        T: Simulation duration
        n_trials: Number of trials with different initial conditions

    Returns:
        rho_mean: Mean empirical stability margin
        rho_std: Standard deviation across trials
    """
    sim = DelayedSwingEquationSimulator(dt=dt, T=T)
    rho_values = []

    for trial in range(n_trials):
        trajectory = sim.simulate(
            A_sys=A_sys,
            B_delay=B_delay,
            tau_vec=tau_vec,
            seed=trial * 42 + 7,
        )
        rho = sim.extract_decay_rate(trajectory)
        rho_values.append(rho)

    return float(np.mean(rho_values)), float(np.std(rho_values))


def build_delay_coupling_matrix(
    A_sys: np.ndarray,
    n_gen: int,
    coupling_strength: float = 0.1,
) -> np.ndarray:
    """
    Build the delay coupling matrix B_delay for the swing equation.

    The delayed term represents the effect of delayed control signals
    on the generator dynamics: B_delay captures how delayed mechanical
    power input P_m(t-tau) affects the swing equation.

    Args:
        A_sys: System matrix [2n, 2n]
        n_gen: Number of generators
        coupling_strength: Coupling strength (relates to K_i)

    Returns:
        B_delay: Delay coupling matrix [2n, 2n]
    """
    n_states = A_sys.shape[0]
    B_delay = np.zeros((n_states, n_states))

    # The delayed term affects the omega (frequency) states
    # d(omega_i)/dt += (coupling_strength / M_i) * delta_j(t - tau)
    # This represents the delayed power system interconnection
    for i in range(n_gen):
        # Cross-coupling through delayed states
        # omega_i is affected by delayed delta_j through the network
        for j in range(n_gen):
            if i != j:
                # Delayed synchronizing torque coefficient
                B_delay[n_gen + i, j] = -coupling_strength / n_gen

    return B_delay
