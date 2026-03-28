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
from controller import LQRController, make_controlled_derivatives, snap_to_equilibrium

# =============================================================================
# Visualization parameters
# =============================================================================
TRAIL_LENGTH = 2000   # number of trail points to keep per bob


# =============================================================================
# Initial conditions  (hardcoded)
# =============================================================================

# θ₁ = θ₂ = 180° (both rods upright), ω₁ = 0 rad/s, ω₂ = 1 rad/s
STATE0 = np.array([np.pi, 0.0, np.pi, 1.0])


def get_controller_target():
    """Ask for stabilization target; Enter = no controller."""
    print("\n  LQR target — valid: (0°,0°) (180°,0°) (0°,180°) (180°,180°)")
    print("  Press Enter to run without controller.\n")
    try:
        raw1 = input("  Target θ₁ [deg, default=180]: ").strip()
        if raw1 == "":
            return None, None
        t1 = snap_to_equilibrium(float(raw1))
        raw2 = input("  Target θ₂ [deg, default=180]: ").strip()
        t2 = snap_to_equilibrium(float(raw2) if raw2 else 180.0)
        print(f"  → target ({t1:.0f}°, {t2:.0f}°)")
        return t1, t2
    except (ValueError, EOFError):
        return None, None


# =============================================================================
# Animation
# =============================================================================

def run_simulation(state0, controller=None):
    """
    Launch the real-time animated visualization.

    Integration happens live in chunks (not pre-computed). The window
    stays open until the user closes it.

    Parameters
    ----------
    state0 : ndarray, shape (4,)
        Initial state.
    controller : LQRController or None
        If provided, the LQR torque is injected at every integrator sub-step
        via a closure over derivatives().  None = free (uncontrolled) simulation.
    """
    # ---- Choose derivatives function ----
    if controller is not None:
        deriv_fn = make_controlled_derivatives(controller)
    else:
        deriv_fn = None   # integrate_chunk falls back to plain derivatives()

    # ---- Mutable simulation state shared via closure ----
    sim = {
        "state": state0.copy(),
        "time": 0.0,
        "buffer_t": [],
        "buffer_y": [],
        "buf_idx": 0,
    }

    # Trail storage
    trail1_x = deque(maxlen=TRAIL_LENGTH)
    trail1_y = deque(maxlen=TRAIL_LENGTH)
    trail2_x = deque(maxlen=TRAIL_LENGTH)
    trail2_y = deque(maxlen=TRAIL_LENGTH)

    def refill_buffer():
        """Integrate the next chunk and fill the frame buffer."""
        t_arr, y_arr = integrate_chunk(sim["state"], sim["time"],
                                       deriv_fn=deriv_fn)
        # Skip first point if continuing (avoids duplicate with previous chunk end)
        start = 1 if sim["time"] > 0 else 0
        sim["buffer_t"] = t_arr[start:].tolist()
        sim["buffer_y"] = y_arr[start:].tolist()
        sim["buf_idx"] = 0
        sim["state"] = y_arr[-1].copy()
        sim["time"] = t_arr[-1]

    refill_buffer()  # pre-fill

    # ---- Set up matplotlib figure ----
    has_ctrl = controller is not None
    title = (
        f"Double Pendulum — LQR  target ({controller.theta1_target_deg:.0f}°,"
        f" {controller.theta2_target_deg:.0f}°)"
        if has_ctrl else
        "Double Pendulum — Free Simulation (friction on)"
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    lim = (L1 + L2) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Target marker (shown only when controller is active)
    if has_ctrl:
        from physics import positions_from_state as _pos
        _tgt = np.array([controller.theta1_eq, 0.0,
                         controller.theta2_eq, 0.0])
        _tx1, _ty1, _tx2, _ty2 = _pos(_tgt)
        ax.plot([0, _tx1, _tx2], [0, _ty1, _ty2],
                "--", color="green", linewidth=1.2, alpha=0.5, zorder=1,
                label="target")
        ax.plot(_tx2, _ty2, "x", color="green", markersize=10,
                markeredgewidth=2, zorder=3)

    # Static elements
    ax.plot(0, 0, "ko", markersize=6, zorder=5)   # pivot

    # Dynamic elements (initially empty)
    rod_line, = ax.plot([], [], "k-", linewidth=2, zorder=4)
    bob1_dot, = ax.plot([], [], "o", color="royalblue", markersize=12,
                        markeredgecolor="black", zorder=5)
    bob2_dot, = ax.plot([], [], "o", color="crimson", markersize=12,
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
    ctrl_text = ax.text(0.98, 0.98, "", transform=ax.transAxes, fontsize=9,
                        verticalalignment="top", horizontalalignment="right",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="lightgreen", alpha=0.8))

    artists = (rod_line, bob1_dot, bob2_dot, trail1_line, trail2_line,
               time_text, energy_text, ctrl_text)

    def init():
        for a in artists:
            if hasattr(a, "set_data"):
                a.set_data([], [])
            else:
                a.set_text("")
        return artists

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
        bob1_dot.set_data([x1], [y1])
        bob2_dot.set_data([x2], [y2])

        # Trails
        trail1_x.append(x1); trail1_y.append(y1)
        trail2_x.append(x2); trail2_y.append(y2)
        trail1_line.set_data(list(trail1_x), list(trail1_y))
        trail2_line.set_data(list(trail2_x), list(trail2_y))

        # Energy (no drift warning — friction dissipates energy intentionally)
        T, V, E = compute_energy(state_now)

        time_text.set_text(f"t = {t_now:8.3f} s")
        energy_text.set_text(
            f"KE  = {T:+10.5f} J\n"
            f"PE  = {V:+10.5f} J\n"
            f"Tot = {E:+10.5f} J"
        )

        # Controller readout
        if has_ctrl:
            tau = controller.last_tau
            import numpy as _np
            phi1 = float(_np.degrees(state_now[0]) - controller.theta1_target_deg)
            phi2 = float(_np.degrees(state_now[2]) - controller.theta2_target_deg)
            # wrap to [-180, 180]
            phi1 = (phi1 + 180) % 360 - 180
            phi2 = (phi2 + 180) % 360 - 180
            ctrl_text.set_text(
                f"LQR τ₁ = {tau:+7.2f} N·m\n"
                f"φ₁   = {phi1:+7.2f}°\n"
                f"φ₂   = {phi2:+7.2f}°"
            )
        else:
            ctrl_text.set_text("controller: OFF")

        return artists

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
    state0 = STATE0.copy()
    print(f"  θ₁={np.degrees(state0[0]):.0f}°  ω₁={state0[1]:.2f} rad/s  "
          f"θ₂={np.degrees(state0[2]):.0f}°  ω₂={state0[3]:.2f} rad/s")

    t1_target, t2_target = get_controller_target()

    controller = None
    if t1_target is not None:
        try:
            controller = LQRController(t1_target, t2_target)
        except ValueError as exc:
            print(f"  [Controller] ERROR: {exc} — running without controller.")

    run_simulation(state0, controller=controller)


if __name__ == "__main__":
    main()
