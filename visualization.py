#!/usr/bin/env python3
"""
Double Pendulum Visualization
==============================

Real-time animation of the double pendulum using matplotlib.
Physics are imported from physics.py — this file handles only:
  - Interactive CLI for initial conditions
  - matplotlib animation loop
  - Trail rendering, energy readout, time display

Run with:  python visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

from physics import (
    L1, L2, M1, M2, G,
    CHUNK_TIME, DT_FRAME,
    derivatives, compute_energy, positions_from_state, integrate_chunk,
)

# =============================================================================
# Visualization parameters
# =============================================================================
TRAIL_LENGTH = 2000   # number of trail points to keep per bob


# =============================================================================
# Interactive CLI for initial conditions
# =============================================================================

def get_initial_conditions():
    """
    Prompt the user for initial angles and angular velocities.

    Returns
    -------
    state0 : ndarray, shape (4,)
        [theta_1, omega_1, theta_2, omega_2] in radians and rad/s.
    """
    print("=" * 60)
    print("  DOUBLE PENDULUM SIMULATION")
    print("  Full nonlinear Lagrangian dynamics")
    print("=" * 60)
    print()
    print("  Angle convention (absolute, from downward vertical):")
    print()
    print("          pivot")
    print("            |      <- 180° (up, unstable equilibrium)")
    print("            |")
    print("      ------+------  <- 90° (horizontal)")
    print("            |")
    print("            |      <- 0° (down, resting position)")
    print("            o  bob")
    print()
    print("  Both θ₁ and θ₂ are measured from the downward vertical.")
    print("  θ₂ is absolute (NOT relative to rod 1).")
    print()
    print("-" * 60)
    print("  Press Enter with no input to load a classic chaotic preset:")
    print("    θ₁ = 120°, θ₂ = -20°, ω₁ = 0°/s, ω₂ = 0°/s")
    print("-" * 60)
    print()

    try:
        raw = input("  θ₁ [degrees, default=120]: ").strip()
        if raw == "":
            theta1_deg, theta2_deg = 120.0, -20.0
            omega1_deg, omega2_deg = 0.0, 0.0
            print(f"\n  Using preset: θ₁={theta1_deg}°, θ₂={theta2_deg}°, "
                  f"ω₁={omega1_deg}°/s, ω₂={omega2_deg}°/s")
        else:
            theta1_deg = float(raw)
            theta2_deg = float(input("  θ₂ [degrees, default=-20]: ").strip() or "-20")
            omega1_deg = float(input("  ω₁ [degrees/s, default=0]: ").strip() or "0")
            omega2_deg = float(input("  ω₂ [degrees/s, default=0]: ").strip() or "0")
    except (ValueError, EOFError):
        print("  Invalid input — using preset.")
        theta1_deg, theta2_deg = 120.0, -20.0
        omega1_deg, omega2_deg = 0.0, 0.0

    # ---------------------------------------------------------------
    # Convert degrees -> radians.
    #
    # ANGLE CONVENTION NOTE:
    # User angles are "from downward vertical". The EOM use the same
    # convention, so this is a direct deg->rad conversion — no offset.
    # This 1:1 mapping is intentional, not an oversight.
    # ---------------------------------------------------------------
    theta1 = np.radians(theta1_deg)
    theta2 = np.radians(theta2_deg)
    omega1 = np.radians(omega1_deg)
    omega2 = np.radians(omega2_deg)

    state0 = np.array([theta1, omega1, theta2, omega2])

    print(f"\n  Internal state: θ₁={theta1:.4f} rad, ω₁={omega1:.4f} rad/s")
    print(f"                  θ₂={theta2:.4f} rad, ω₂={omega2:.4f} rad/s\n")

    return state0


# =============================================================================
# Animation
# =============================================================================

def run_simulation(state0):
    """
    Launch the real-time animated visualization.

    Integration happens live in chunks (not pre-computed). The window
    stays open until the user closes it.

    Parameters
    ----------
    state0 : ndarray, shape (4,)
        Initial state.
    """
    # ---- Mutable simulation state shared via closure ----
    sim = {
        "state": state0.copy(),
        "time": 0.0,
        "buffer_t": [],
        "buffer_y": [],
        "buf_idx": 0,
    }

    _, _, E0 = compute_energy(state0)
    sim["E0"] = E0

    # Trail storage
    trail1_x = deque(maxlen=TRAIL_LENGTH)
    trail1_y = deque(maxlen=TRAIL_LENGTH)
    trail2_x = deque(maxlen=TRAIL_LENGTH)
    trail2_y = deque(maxlen=TRAIL_LENGTH)

    energy_warned = [False]

    def refill_buffer():
        """Integrate the next chunk and fill the frame buffer."""
        t_arr, y_arr = integrate_chunk(sim["state"], sim["time"])
        # Skip first point if continuing (avoids duplicate with previous chunk end)
        start = 1 if sim["time"] > 0 else 0
        sim["buffer_t"] = t_arr[start:].tolist()
        sim["buffer_y"] = y_arr[start:].tolist()
        sim["buf_idx"] = 0
        sim["state"] = y_arr[-1].copy()
        sim["time"] = t_arr[-1]

    refill_buffer()  # pre-fill

    # ---- Set up matplotlib figure ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    lim = (L1 + L2) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Double Pendulum — Full Nonlinear Simulation")
    ax.grid(True, alpha=0.3)

    # Static elements
    ax.plot(0, 0, "ko", markersize=6, zorder=5)   # pivot

    # Dynamic elements (initially empty)
    rod_line, = ax.plot([], [], "k-", linewidth=2, zorder=4)
    bob1, = ax.plot([], [], "o", color="royalblue", markersize=12,
                    markeredgecolor="black", zorder=5)
    bob2, = ax.plot([], [], "o", color="crimson", markersize=12,
                    markeredgecolor="black", zorder=5)
    trail1_line, = ax.plot([], [], "-", color="royalblue", linewidth=0.8,
                           alpha=0.6, zorder=2)
    trail2_line, = ax.plot([], [], "-", color="crimson", linewidth=0.8,
                           alpha=0.6, zorder=2)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=10,
                        verticalalignment="top", fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="wheat", alpha=0.8))
    energy_text = ax.text(0.02, 0.82, "", transform=ax.transAxes, fontsize=9,
                          verticalalignment="top", fontfamily="monospace",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="lightyellow", alpha=0.8))

    def init():
        rod_line.set_data([], [])
        bob1.set_data([], [])
        bob2.set_data([], [])
        trail1_line.set_data([], [])
        trail2_line.set_data([], [])
        time_text.set_text("")
        energy_text.set_text("")
        return rod_line, bob1, bob2, trail1_line, trail2_line, time_text, energy_text

    def update(frame):
        # Refill if buffer exhausted
        if sim["buf_idx"] >= len(sim["buffer_t"]):
            refill_buffer()

        idx = sim["buf_idx"]
        t_now = sim["buffer_t"][idx]
        state_now = np.array(sim["buffer_y"][idx])
        sim["buf_idx"] = idx + 1

        # Cartesian positions
        x1, y1, x2, y2 = positions_from_state(state_now)

        # Rods: pivot -> bob1 -> bob2
        rod_line.set_data([0, x1, x2], [0, y1, y2])
        bob1.set_data([x1], [y1])
        bob2.set_data([x2], [y2])

        # Trails
        trail1_x.append(x1); trail1_y.append(y1)
        trail2_x.append(x2); trail2_y.append(y2)
        trail1_line.set_data(list(trail1_x), list(trail1_y))
        trail2_line.set_data(list(trail2_x), list(trail2_y))

        # Energy
        T, V, E = compute_energy(state_now)
        drift_pct = abs((E - sim["E0"]) / sim["E0"]) * 100.0 if sim["E0"] != 0 else 0.0

        if drift_pct > 0.1 and not energy_warned[0]:
            energy_warned[0] = True
            print(f"  WARNING: Total energy drift exceeded 0.1% at t={t_now:.2f}s "
                  f"(drift={drift_pct:.4f}%)")

        time_text.set_text(f"t = {t_now:8.3f} s")
        energy_text.set_text(
            f"KE  = {T:+10.5f} J\n"
            f"PE  = {V:+10.5f} J\n"
            f"Tot = {E:+10.5f} J\n"
            f"dE  = {drift_pct:8.6f} %"
        )

        return rod_line, bob1, bob2, trail1_line, trail2_line, time_text, energy_text

    _anim = FuncAnimation(
        fig, update, init_func=init,
        interval=DT_FRAME * 1000,
        blit=True,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    state0 = get_initial_conditions()

    print("=" * 60)
    print("  Starting simulation...")
    print("  Close the matplotlib window to stop.")
    print("=" * 60)

    _, _, E0 = compute_energy(state0)
    print(f"\n  Initial total energy: {E0:.6f} J\n")

    run_simulation(state0)
    print("  Simulation ended.")


if __name__ == "__main__":
    main()
