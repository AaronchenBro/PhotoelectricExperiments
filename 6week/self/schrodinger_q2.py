#!/usr/bin/env python3
"""Q2 finite-difference solver for a 1D finite quantum well.

The assignment asks for the energy levels and wavefunctions of the reference
quantum-well figure using m* = 0.067 m0.  The default parameters are tuned to
the Davies-style figure shown in the handout: a 10 nm well, 0.30 eV barriers,
and a -20 to 20 nm plotting window.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal

try:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/week6_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from exc


# hbar^2 / (2 m0), in eV nm^2.
HBAR2_OVER_2M0_EV_NM2 = 0.0380998212


@dataclass(frozen=True)
class WellResult:
    z_nm: np.ndarray
    potential_ev: np.ndarray
    energies_ev: np.ndarray
    wavefunctions: np.ndarray
    mass_ratio: float
    well_width_nm: float
    barrier_ev: float


def finite_square_well(z_nm: np.ndarray, well_width_nm: float, barrier_ev: float) -> np.ndarray:
    """Return V(z) for a centered finite square well."""
    half_width = well_width_nm / 2.0
    return np.where(np.abs(z_nm) <= half_width, 0.0, barrier_ev)


def solve_well(
    *,
    mass_ratio: float = 0.067,
    well_width_nm: float = 10.0,
    barrier_ev: float = 0.30,
    z_min_nm: float = -20.0,
    z_max_nm: float = 20.0,
    grid_points: int = 2401,
) -> WellResult:
    """Solve the time-independent Schrodinger equation by finite differences."""
    if grid_points < 101:
        raise ValueError("grid_points should be at least 101 for a stable-looking plot.")
    if mass_ratio <= 0:
        raise ValueError("mass_ratio must be positive.")
    if well_width_nm <= 0 or barrier_ev <= 0:
        raise ValueError("well_width_nm and barrier_ev must be positive.")

    z_nm = np.linspace(z_min_nm, z_max_nm, grid_points)
    dz_nm = z_nm[1] - z_nm[0]
    potential_ev = finite_square_well(z_nm, well_width_nm, barrier_ev)

    kinetic = HBAR2_OVER_2M0_EV_NM2 / mass_ratio / dz_nm**2
    # Dirichlet boundaries: psi(z_min) = psi(z_max) = 0.
    diagonal = 2.0 * kinetic + potential_ev[1:-1]
    off_diagonal = -kinetic * np.ones(grid_points - 3)

    energies_ev, interior_vecs = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="v",
        select_range=(0.0, barrier_ev - 1e-12),
    )

    wavefunctions = np.zeros((len(energies_ev), grid_points))
    for state_index, psi_inner in enumerate(interior_vecs.T):
        psi = np.zeros(grid_points)
        psi[1:-1] = psi_inner
        norm = np.sqrt(np.trapezoid(psi**2, z_nm))
        psi /= norm
        # Fix sign for stable plots: make the largest lobe positive.
        if psi[np.argmax(np.abs(psi))] < 0:
            psi *= -1.0
        wavefunctions[state_index] = psi

    return WellResult(
        z_nm=z_nm,
        potential_ev=potential_ev,
        energies_ev=energies_ev,
        wavefunctions=wavefunctions,
        mass_ratio=mass_ratio,
        well_width_nm=well_width_nm,
        barrier_ev=barrier_ev,
    )


def scaled_wavefunction(result: WellResult, state_index: int, amplitude_ev: float) -> np.ndarray:
    """Return psi scaled in eV units and shifted to its eigenenergy."""
    psi = result.wavefunctions[state_index]
    psi = psi / np.max(np.abs(psi))
    return result.energies_ev[state_index] + amplitude_ev * psi


def plot_overlay(result: WellResult, output_path: Path, states: int = 3) -> None:
    """Plot V(z), the first energy levels, and energy-shifted wavefunctions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.plot(result.z_nm, result.potential_ev, color="black", lw=1.8, label="V(z)")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    n_states = min(states, len(result.energies_ev))
    for i in range(n_states):
        energy = result.energies_ev[i]
        ax.hlines(energy, result.z_nm[0], result.z_nm[-1], colors="0.35", linestyles="--", lw=0.9)
        ax.plot(
            result.z_nm,
            scaled_wavefunction(result, i, amplitude_ev=0.075),
            color=colors[i % len(colors)],
            lw=2.0,
            label=f"n={i + 1}, E={energy:.3f} eV",
        )
        ax.text(17.0, energy + 0.004, f"n = {i + 1}", ha="right", va="bottom", fontsize=10)

    ax.annotate(
        "$V_0$",
        xy=(-14.0, result.barrier_ev),
        xytext=(-14.0, 0.02),
        arrowprops={"arrowstyle": "<->", "lw": 1.0, "color": "0.2"},
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set(
        xlim=(result.z_nm[0], result.z_nm[-1]),
        ylim=(-0.015, result.barrier_ev + 0.065),
        xlabel="z / nm",
        ylabel="E / eV",
        title="Finite quantum well eigenstates from finite differences",
    )
    ax.grid(alpha=0.20, lw=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_single_state(result: WellResult, state_index: int, output_path: Path) -> None:
    """Plot one normalized wavefunction with its eigenenergy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    energy = result.energies_ev[state_index]
    psi = result.wavefunctions[state_index] / np.max(np.abs(result.wavefunctions[state_index]))

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    ax1.plot(result.z_nm, result.potential_ev, color="black", lw=1.7, label="V(z)")
    ax1.hlines(energy, result.z_nm[0], result.z_nm[-1], colors="0.35", linestyles="--", lw=0.9)
    ax1.set(xlabel="z / nm", ylabel="E / eV", ylim=(-0.015, result.barrier_ev + 0.045))
    ax1.grid(alpha=0.20, lw=0.6)

    ax2 = ax1.twinx()
    ax2.plot(result.z_nm, psi, color="tab:blue", lw=2.0, label=f"$\\psi_{state_index + 1}(z)$")
    ax2.fill_between(result.z_nm, 0.0, psi, color="tab:blue", alpha=0.13)
    ax2.set(ylabel="normalized wavefunction", ylim=(-1.15, 1.15))

    ax1.set_title(f"State n={state_index + 1}: E = {energy:.4f} eV")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_summary(result: WellResult, output_path: Path) -> None:
    """Write eigenenergy and simple parity diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "state_n",
                "energy_eV",
                "barrier_eV",
                "well_width_nm",
                "effective_mass_m0",
            ]
        )
        for i, energy in enumerate(result.energies_ev, start=1):
            writer.writerow([i, f"{energy:.8f}", result.barrier_ev, result.well_width_nm, result.mass_ratio])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve and plot the Q2 finite quantum well.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--mass-ratio", type=float, default=0.067, help="effective mass in units of m0")
    parser.add_argument("--well-width-nm", type=float, default=10.0)
    parser.add_argument("--barrier-ev", type=float, default=0.30)
    parser.add_argument("--z-min-nm", type=float, default=-20.0)
    parser.add_argument("--z-max-nm", type=float, default=20.0)
    parser.add_argument("--grid-points", type=int, default=2401)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = solve_well(
        mass_ratio=args.mass_ratio,
        well_width_nm=args.well_width_nm,
        barrier_ev=args.barrier_ev,
        z_min_nm=args.z_min_nm,
        z_max_nm=args.z_max_nm,
        grid_points=args.grid_points,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_overlay(result, args.output_dir / "q2_quantum_well_overlay.png")
    for state_index in range(min(3, len(result.energies_ev))):
        plot_single_state(result, state_index, args.output_dir / f"q2_state_n{state_index + 1}.png")
    write_summary(result, args.output_dir / "q2_bound_state_summary.csv")

    print("Bound-state energies:")
    for i, energy in enumerate(result.energies_ev, start=1):
        print(f"  n={i}: {energy:.6f} eV")
    print(f"Figures written to: {args.output_dir}")


if __name__ == "__main__":
    main()
