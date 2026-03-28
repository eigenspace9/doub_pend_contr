#!/usr/bin/env python3
"""
Double Pendulum Visualization
==============================

Real-time animation of the double pendulum using matplotlib.
Physics are imported from physics.py — this file handles only:
  - Interactive CLI for initial conditions
  - matplotlib animation loop
  - Trail rendering, energy readout, time display
  - Torque and power time-series plots

Run with:  python visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from collections import deque

from physics import (
    L1, L2, M1, M2, G,
    CHUNK_TIME, DT_FRAME,
    derivatives, compute_energy, positions_from_state, integrate_chunk,
)
from matplotlib.widgets import Button
from controller import (
    LQRController, HybridController,
    snap_to_equilibrium, TAU_MAX,
)

# =============================================================================
# Visualization parameters
# =============================================================================
TRAIL_LENGTH  = 2000   # number of trail points to keep per bob
HISTORY_LEN   = 500    # rolling torque/power history (~10 s at 50 fps)
PUSH_TORQUE   = 15.0   # sustained push torque [N·m]
PUSH_DURATION = 0.5    # how long a push lasts [s]


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
    has_ctrl = controller is not None

    # ---- Mutable simulation state shared via closure ----
    sim = {
        "state": state0.copy(),
        "time": 0.0,
        "buffer_t": [],
        "buffer_y": [],
        "buf_idx": 0,
        "push_torque_until": 0.0,   # timestamp when push torque expires
        "push_torque_value": 0.0,   # push torque magnitude [N·m]
    }

    # ---- Choose derivatives function (push-aware) ----
    def deriv_fn(t, state):
        torque = np.array([0.0, 0.0])
        if controller is not None:
            torque = controller(t, state)
        if t < sim["push_torque_until"]:
            torque[0] += sim["push_torque_value"]
        return derivatives(t, state, torque=torque)

    # Trail storage
    trail1_x = deque(maxlen=TRAIL_LENGTH)
    trail1_y = deque(maxlen=TRAIL_LENGTH)
    trail2_x = deque(maxlen=TRAIL_LENGTH)
    trail2_y = deque(maxlen=TRAIL_LENGTH)

    # Torque / power history
    t_hist   = deque(maxlen=HISTORY_LEN)
    tau_hist = deque(maxlen=HISTORY_LEN)
    pwr_hist = deque(maxlen=HISTORY_LEN)

    def refill_buffer():
        """Integrate the next chunk and fill the frame buffer."""
        MAX_RETRIES = 3
        chunk = CHUNK_TIME
        for attempt in range(MAX_RETRIES):
            try:
                t_arr, y_arr = integrate_chunk(sim["state"], sim["time"],
                                               chunk_time=chunk, deriv_fn=deriv_fn)
                start = 1 if sim["time"] > 0 else 0
                sim["buffer_t"] = t_arr[start:].tolist()
                sim["buffer_y"] = y_arr[start:].tolist()
                sim["buf_idx"] = 0
                sim["state"] = y_arr[-1].copy()
                sim["time"] = t_arr[-1]
                return
            except RuntimeError as exc:
                print(f"  [Integration] Retry {attempt+1}/{MAX_RETRIES}: {exc}")
                chunk *= 0.5

        # All retries failed — clamp velocities and try once more
        print("  [Integration] Clamping state and retrying...")
        MAX_OMEGA = 50.0
        sim["state"][1] = np.clip(sim["state"][1], -MAX_OMEGA, MAX_OMEGA)
        sim["state"][3] = np.clip(sim["state"][3], -MAX_OMEGA, MAX_OMEGA)
        try:
            t_arr, y_arr = integrate_chunk(sim["state"], sim["time"],
                                           chunk_time=0.1, deriv_fn=deriv_fn)
            start = 1 if sim["time"] > 0 else 0
            sim["buffer_t"] = t_arr[start:].tolist()
            sim["buffer_y"] = y_arr[start:].tolist()
            sim["buf_idx"] = 0
            sim["state"] = y_arr[-1].copy()
            sim["time"] = t_arr[-1]
        except RuntimeError:
            # Last resort: hold state and advance time
            print("  [Integration] FAILED — holding state, advancing time.")
            dt = DT_FRAME
            sim["buffer_t"] = [sim["time"] + dt]
            sim["buffer_y"] = [sim["state"].tolist()]
            sim["buf_idx"] = 0
            sim["time"] += dt

    refill_buffer()  # pre-fill

    # ---- Set up matplotlib figure ----
    title = (
        f"Double Pendulum — Swing-up + LQR  target"
        f" ({controller.theta1_target_deg:.0f}°,"
        f" {controller.theta2_target_deg:.0f}°)"
        if has_ctrl else
        "Double Pendulum — Free Simulation (friction on)"
    )

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(title, fontsize=11)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[1.1, 1],
                           hspace=0.55, wspace=0.40)

    ax      = fig.add_subplot(gs[:, 0])   # pendulum animation (full left column)
    ax_tau  = fig.add_subplot(gs[0, 1])   # torque time series
    ax_pwr  = fig.add_subplot(gs[1, 1])   # power time series

    # ---- Pendulum axes ----
    ax.set_aspect("equal")
    lim = (L1 + L2) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)

    # Target marker
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

    ax.plot(0, 0, "ko", markersize=6, zorder=5)   # pivot

    rod_line,  = ax.plot([], [], "k-", linewidth=2, zorder=4)
    bob1_dot,  = ax.plot([], [], "o", color="royalblue", markersize=12,
                         markeredgecolor="black", zorder=5)
    bob2_dot,  = ax.plot([], [], "o", color="crimson", markersize=12,
                         markeredgecolor="black", zorder=5)
    trail1_ln, = ax.plot([], [], "-", color="royalblue", linewidth=0.8,
                         alpha=0.6, zorder=2)
    trail2_ln, = ax.plot([], [], "-", color="crimson", linewidth=0.8,
                         alpha=0.6, zorder=2)

    time_text   = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=10,
                          verticalalignment="top", fontfamily="monospace",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="wheat", alpha=0.8))
    energy_text = ax.text(0.02, 0.82, "", transform=ax.transAxes, fontsize=9,
                          verticalalignment="top", fontfamily="monospace",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="lightyellow", alpha=0.8))
    ctrl_text   = ax.text(0.98, 0.98, "", transform=ax.transAxes, fontsize=9,
                          verticalalignment="top", horizontalalignment="right",
                          fontfamily="monospace",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="lightgreen", alpha=0.8))

    # ---- Torque axes ----
    ax_tau.set_xlim(0, HISTORY_LEN * DT_FRAME)
    ax_tau.set_ylim(-TAU_MAX * 1.1, TAU_MAX * 1.1)
    ax_tau.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax_tau.axhline( TAU_MAX, color="salmon", linewidth=0.8, linestyle=":")
    ax_tau.axhline(-TAU_MAX, color="salmon", linewidth=0.8, linestyle=":")
    ax_tau.set_xlabel("time window [s]")
    ax_tau.set_ylabel("τ₁ [N·m]")
    ax_tau.set_title("Joint-1 torque")
    ax_tau.grid(True, alpha=0.3)
    tau_line, = ax_tau.plot([], [], color="darkorange", linewidth=1.2)

    # ---- Power axes ----
    PWR_LIM = TAU_MAX * 8.0   # generous initial y-limit
    ax_pwr.set_xlim(0, HISTORY_LEN * DT_FRAME)
    ax_pwr.set_ylim(-PWR_LIM, PWR_LIM)
    ax_pwr.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax_pwr.set_xlabel("time window [s]")
    ax_pwr.set_ylabel("P [W]")
    ax_pwr.set_title("Joint-1 power  (τ₁ · ω₁)")
    ax_pwr.grid(True, alpha=0.3)
    pwr_line, = ax_pwr.plot([], [], color="mediumpurple", linewidth=1.2)

    # ---- Push button ----
    # Leaves a strip at the bottom of the figure for the button.
    # We call tight_layout with rect later to keep axes clear of it.
    ax_push = fig.add_axes([0.02, 0.01, 0.16, 0.055])
    push_btn = Button(ax_push, 'Push  [P / Space]',
                      color='lightcoral', hovercolor='tomato')

    def apply_push(event=None):
        """Queue a sustained torque push on rod 1."""
        sim["push_torque_until"] = sim["time"] + PUSH_DURATION
        sim["push_torque_value"] = PUSH_TORQUE
        # Clear buffer so integration restarts with push torque active
        sim["buffer_t"] = []
        sim["buffer_y"] = []
        sim["buf_idx"] = 0

    push_btn.on_clicked(apply_push)
    fig.canvas.mpl_connect(
        'key_press_event',
        lambda e: apply_push() if e.key in ('p', ' ') else None
    )

    artists = (rod_line, bob1_dot, bob2_dot, trail1_ln, trail2_ln,
               time_text, energy_text, ctrl_text,
               tau_line, pwr_line)

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
        if len(sim["buffer_t"]) == 0:
            return artists

        idx        = sim["buf_idx"]
        t_now      = sim["buffer_t"][idx]
        state_now  = np.array(sim["buffer_y"][idx])
        sim["buf_idx"] = idx + 1

        # Cartesian positions
        x1, y1, x2, y2 = positions_from_state(state_now)

        # Rods
        rod_line.set_data([0, x1, x2], [0, y1, y2])
        bob1_dot.set_data([x1], [y1])
        bob2_dot.set_data([x2], [y2])

        # Trails
        trail1_x.append(x1); trail1_y.append(y1)
        trail2_x.append(x2); trail2_y.append(y2)
        trail1_ln.set_data(list(trail1_x), list(trail1_y))
        trail2_ln.set_data(list(trail2_x), list(trail2_y))

        # Energy
        T, V, E = compute_energy(state_now)
        time_text.set_text(f"t = {t_now:8.3f} s")
        energy_text.set_text(
            f"KE  = {T:+10.5f} J\n"
            f"PE  = {V:+10.5f} J\n"
            f"Tot = {E:+10.5f} J"
        )

        # Controller readout + torque/power history
        theta1, omega1, theta2, omega2 = state_now
        if has_ctrl:
            tau = controller.last_tau
            phi1 = (np.degrees(theta1) - controller.theta1_target_deg + 180) % 360 - 180
            phi2 = (np.degrees(theta2) - controller.theta2_target_deg + 180) % 360 - 180
            mode = getattr(controller, 'mode', 'lqr').upper()
            ctrl_text.set_text(
                f"[{mode}] τ₁ = {tau:+7.2f} N·m\n"
                f"φ₁   = {phi1:+7.2f}°\n"
                f"φ₂   = {phi2:+7.2f}°"
            )
        else:
            tau = 0.0
            ctrl_text.set_text("controller: OFF")

        power = tau * omega1

        # Rolling time axis: shift so latest point is at right edge
        t_hist.append(t_now)
        tau_hist.append(tau)
        pwr_hist.append(power)

        t_arr  = np.array(t_hist)
        t_rel  = t_arr - t_arr[-1] + HISTORY_LEN * DT_FRAME  # shift to window

        tau_line.set_data(t_rel, list(tau_hist))
        pwr_line.set_data(t_rel, list(pwr_hist))

        # Auto-scale power y-axis if needed (no blit issue since limits reset each frame)
        if len(pwr_hist) > 1:
            p_max = max(abs(p) for p in pwr_hist) * 1.2 or PWR_LIM
            if p_max > PWR_LIM:
                ax_pwr.set_ylim(-p_max, p_max)

        return artists

    _anim = FuncAnimation(
        fig, update, init_func=init,
        interval=DT_FRAME * 1000,
        blit=False,          # False so axis rescaling works
        cache_frame_data=False,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])   # leave bottom strip for push button
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    t1_target, t2_target = get_controller_target()

    state0 = np.array([np.pi / 2, 0.0, np.pi / 2, 0.0])
    print(f"  θ₁={np.degrees(state0[0]):.0f}°  ω₁={state0[1]:.2f} rad/s  "
          f"θ₂={np.degrees(state0[2]):.0f}°  ω₂={state0[3]:.2f} rad/s")

    controller = None
    if t1_target is not None:
        try:
            controller = HybridController(t1_target, t2_target)
        except ValueError as exc:
            print(f"  [Controller] ERROR: {exc} — running without controller.")

    run_simulation(state0, controller=controller)


if __name__ == "__main__":
    main()
