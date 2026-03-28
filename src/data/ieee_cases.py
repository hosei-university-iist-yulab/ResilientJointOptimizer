#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/21/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
IEEE Test Case Loader

Loads standard IEEE power system test cases (14, 39, 118, 300 bus)
using pandapower library.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

try:
    import pandapower as pp
    import pandapower.networks as pn
except ImportError:
    raise ImportError("pandapower is required. Install with: pip install pandapower")


@dataclass
class PowerSystemState:
    """Container for power system state variables."""

    # Bus quantities
    P: torch.Tensor        # Active power [n_bus]
    Q: torch.Tensor        # Reactive power [n_bus]
    V: torch.Tensor        # Voltage magnitude [n_bus]
    theta: torch.Tensor    # Voltage angle [n_bus]

    # Generator quantities
    P_gen: torch.Tensor    # Generator active power [n_gen]
    Q_gen: torch.Tensor    # Generator reactive power [n_gen]

    # Topology
    edge_index: torch.Tensor   # Line connections [2, n_lines]
    line_impedance: torch.Tensor  # Line impedance [n_lines]

    # Indices
    gen_bus_idx: torch.Tensor  # Generator bus indices [n_gen]
    load_bus_idx: torch.Tensor  # Load bus indices [n_load]

    # Metadata
    n_bus: int
    n_gen: int
    n_line: int


class IEEECaseLoader:
    """
    Loads IEEE standard test cases.

    Supported cases: 4, 6, 9, 14, 30, 39, 57, 118, 300
    """

    SUPPORTED_CASES = {
        39: "case39",           # New England
        57: "case57",
        118: "case118",
        145: "case145",
        300: "case300",
        1354: "case1354pegase", # European grid (PEGASE project)
        1888: "case1888rte",    # French transmission grid (RTE)
        2869: "case2869pegase", # Large European grid (PEGASE project)
    }

    def __init__(self, case_id: int):
        """
        Args:
            case_id: IEEE case number (14, 39, 118, etc.)
        """
        if case_id not in self.SUPPORTED_CASES:
            raise ValueError(
                f"Case {case_id} not supported. "
                f"Available: {list(self.SUPPORTED_CASES.keys())}"
            )

        self.case_id = case_id
        self.case_name = self.SUPPORTED_CASES[case_id]
        self.net = self._load_network()

    def _load_network(self):
        """Load pandapower network."""
        loader_func = getattr(pn, self.case_name)
        net = loader_func()
        return net

    def run_power_flow(self) -> bool:
        """Run AC power flow and return convergence status."""
        try:
            pp.runpp(self.net, algorithm="nr", max_iteration=30)
            return self.net.converged
        except Exception as e:
            print(f"Power flow failed: {e}")
            return False

    def get_state(self) -> PowerSystemState:
        """
        Extract power system state after running power flow.

        Returns:
            PowerSystemState object with all quantities
        """
        if not hasattr(self.net, "res_bus") or self.net.res_bus.empty:
            if not self.run_power_flow():
                raise RuntimeError("Power flow did not converge")

        # Bus quantities
        n_bus = len(self.net.bus)
        P = torch.tensor(self.net.res_bus.p_mw.values, dtype=torch.float32)
        Q = torch.tensor(self.net.res_bus.q_mvar.values, dtype=torch.float32)
        V = torch.tensor(self.net.res_bus.vm_pu.values, dtype=torch.float32)
        theta = torch.tensor(
            self.net.res_bus.va_degree.values * np.pi / 180,  # Convert to radians
            dtype=torch.float32
        )

        # Generator quantities
        n_gen = len(self.net.gen) + len(self.net.ext_grid)  # Include slack

        # Generator power (combine gen and ext_grid)
        P_gen_list = []
        Q_gen_list = []
        gen_bus_list = []

        # External grid (slack bus)
        for idx in self.net.ext_grid.index:
            bus = self.net.ext_grid.at[idx, "bus"]
            P_gen_list.append(self.net.res_ext_grid.at[idx, "p_mw"])
            Q_gen_list.append(self.net.res_ext_grid.at[idx, "q_mvar"])
            gen_bus_list.append(bus)

        # Regular generators
        for idx in self.net.gen.index:
            bus = self.net.gen.at[idx, "bus"]
            P_gen_list.append(self.net.res_gen.at[idx, "p_mw"])
            Q_gen_list.append(self.net.res_gen.at[idx, "q_mvar"])
            gen_bus_list.append(bus)

        P_gen = torch.tensor(P_gen_list, dtype=torch.float32)
        Q_gen = torch.tensor(Q_gen_list, dtype=torch.float32)
        gen_bus_idx = torch.tensor(gen_bus_list, dtype=torch.long)

        # Load bus indices
        load_buses = self.net.load.bus.unique()
        load_bus_idx = torch.tensor(load_buses, dtype=torch.long)

        # Line topology
        from_bus = self.net.line.from_bus.values
        to_bus = self.net.line.to_bus.values
        edge_index = torch.tensor(
            np.array([from_bus, to_bus]),
            dtype=torch.long
        )

        # Line impedance (simplified: use length * r_ohm_per_km)
        r = self.net.line.r_ohm_per_km.values
        x = self.net.line.x_ohm_per_km.values
        length = self.net.line.length_km.values
        z_magnitude = np.sqrt(r**2 + x**2) * length
        line_impedance = torch.tensor(z_magnitude, dtype=torch.float32)

        n_line = len(self.net.line)

        return PowerSystemState(
            P=P, Q=Q, V=V, theta=theta,
            P_gen=P_gen, Q_gen=Q_gen,
            edge_index=edge_index,
            line_impedance=line_impedance,
            gen_bus_idx=gen_bus_idx,
            load_bus_idx=load_bus_idx,
            n_bus=n_bus,
            n_gen=n_gen,
            n_line=n_line,
        )

    def _get_generator_ratings_mw(self) -> np.ndarray:
        """Extract MW ratings for each generator from pandapower data."""
        p_mw_list = []

        # External grid (slack bus)
        for idx in self.net.ext_grid.index:
            p = self.net.ext_grid.at[idx, 'max_p_mw'] if 'max_p_mw' in self.net.ext_grid.columns else 500.0
            if np.isnan(p):
                p = 500.0
            p_mw_list.append(abs(max(p, 10.0)))

        # Regular generators
        for idx in self.net.gen.index:
            p = self.net.gen.at[idx, 'max_p_mw'] if 'max_p_mw' in self.net.gen.columns else 100.0
            if np.isnan(p):
                p = 100.0
            p_mw_list.append(abs(max(p, 10.0)))

        return np.array(p_mw_list, dtype=np.float64)

    def get_generator_dynamics(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute generator-specific inertia H_i and damping D_i from machine ratings.

        Inertia scaling: H_i = 2 + 6 * sqrt(P_i / P_max) seconds
          - Small machines (~10MW):   H ~ 2-3s (gas turbines, small hydro)
          - Large machines (~1000MW): H ~ 6-8s (large thermal, nuclear)

        Damping: D_i = 0.05 * H_i + noise (1-5% of inertia, typical)

        Returns:
            H: Inertia constants [n_gen] in seconds
            D: Damping coefficients [n_gen]
            Ks: Synchronizing torque coefficients [n_gen]
        """
        p_mw = self._get_generator_ratings_mw()
        n = len(p_mw)

        # Inertia: empirical scaling from machine rating
        p_norm = p_mw / p_mw.max()
        H = 2.0 + 6.0 * np.sqrt(p_norm)

        # Damping: proportional to inertia with small variation
        rng = np.random.RandomState(self.case_id)  # Deterministic per case
        D = 0.05 * H + 0.5 * rng.uniform(0.8, 1.2, n)

        # Synchronizing torque coefficient: proportional to active power
        Ks = p_mw / 100.0

        return (
            torch.tensor(H, dtype=torch.float32),
            torch.tensor(D, dtype=torch.float32),
            torch.tensor(Ks, dtype=torch.float32),
        )

    def get_system_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get linearized system matrices A and B for stability analysis.

        Uses generator-specific inertia H_i and damping D_i derived from
        machine MW ratings (not uniform M=5, D=2 for all generators).

        Returns:
            A: System matrix [2*n_gen, 2*n_gen]
            B: Input matrix [2*n_gen, n_gen]
        """
        H, D, Ks = self.get_generator_dynamics()
        n = len(H)
        M = 2.0 * H  # M = 2H in per-unit convention

        # A matrix: [0, I; -Ks/M, -D/M]
        A = torch.zeros(2*n, 2*n)
        A[:n, n:] = torch.eye(n)                   # d(delta)/dt = omega
        A[n:, :n] = -torch.diag(Ks / M)            # d(omega)/dt = -Ks/M * delta
        A[n:, n:] = -torch.diag(D / M)             # d(omega)/dt += -D/M * omega

        # B matrix: control input affects omega
        B = torch.zeros(2*n, n)
        B[n:, :] = torch.diag(1.0 / M)

        return A, B

    def compute_jacobian(self) -> torch.Tensor:
        """
        Compute power flow Jacobian matrix.

        Returns:
            J: Jacobian matrix [2*n_bus, 2*n_bus]
        """
        # Run power flow if needed
        if not hasattr(self.net, "res_bus") or self.net.res_bus.empty:
            self.run_power_flow()

        # Get bus admittance matrix
        from pandapower.pypower.makeYbus import makeYbus
        from pandapower.pd2ppc import _pd2ppc

        ppc, _ = _pd2ppc(self.net)
        Ybus, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])

        # Convert to torch
        Y = torch.tensor(Ybus.toarray(), dtype=torch.complex64)

        # Compute Jacobian elements
        n = len(self.net.bus)
        V = torch.tensor(
            self.net.res_bus.vm_pu.values *
            np.exp(1j * self.net.res_bus.va_degree.values * np.pi / 180),
            dtype=torch.complex64
        )

        # Standard power flow Jacobian
        # J = [J11, J12; J21, J22]
        # J11 = dP/dtheta, J12 = dP/dV
        # J21 = dQ/dtheta, J22 = dQ/dV

        J = torch.zeros(2*n, 2*n)

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements
                    J[i, i] = -V[i].imag * torch.sum(
                        Y[i, :] * V
                    ).real + V[i].real * torch.sum(
                        Y[i, :] * V
                    ).imag
                else:
                    # Off-diagonal
                    J[i, j] = V[i].abs() * V[j].abs() * (
                        Y[i, j].real * torch.sin(V[i].angle() - V[j].angle()) -
                        Y[i, j].imag * torch.cos(V[i].angle() - V[j].angle())
                    )

        return J.real

    def get_eigenvalues(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigenvalues of the system Jacobian.

        Returns:
            eigenvalues: Complex eigenvalues
            lambda_min: Minimum real part (stability indicator)
        """
        A, _ = self.get_system_matrices()
        eigenvalues = torch.linalg.eigvals(A)

        # Minimum real part (most unstable/least stable mode)
        lambda_min = eigenvalues.real.min()

        return eigenvalues, lambda_min


def load_ieee_case(case_id: int) -> IEEECaseLoader:
    """Convenience function to load IEEE case."""
    return IEEECaseLoader(case_id)


# Add load method to IEEECaseLoader for dataset compatibility
def _ieee_case_load(self) -> Dict:
    """
    Load IEEE case and return dict with all data needed for training.

    Returns:
        Dict with keys: n_buses, n_generators, edge_index, V, theta,
        P_load, Q_load, P_gen, Q_gen, lambda_min, gen_buses, impedance_matrix
    """
    state = self.get_state()
    _, lambda_min = self.get_eigenvalues()

    # Build impedance matrix from edges
    n = state.n_bus
    impedance_matrix = torch.ones(n, n) * 1e6  # Large default
    row, col = state.edge_index
    for i, (r, c) in enumerate(zip(row.tolist(), col.tolist())):
        impedance_matrix[r, c] = state.line_impedance[i]
        impedance_matrix[c, r] = state.line_impedance[i]
    impedance_matrix.fill_diagonal_(0)

    return {
        'n_buses': state.n_bus,
        'n_generators': state.n_gen,
        'n_lines': state.n_line,
        'edge_index': state.edge_index,
        'V': state.V,
        'theta': state.theta,
        'P_load': state.P,  # Bus-level power (load is negative)
        'Q_load': state.Q,
        'P_gen': state.P_gen,
        'Q_gen': state.Q_gen,
        'gen_buses': state.gen_bus_idx,
        'lambda_min': lambda_min.item(),
        'impedance_matrix': impedance_matrix,
        'line_impedance': state.line_impedance,
    }


# Monkey-patch the method onto the class
IEEECaseLoader.load = _ieee_case_load


def get_all_cases() -> List[int]:
    """Return list of all supported IEEE cases."""
    return list(IEEECaseLoader.SUPPORTED_CASES.keys())
