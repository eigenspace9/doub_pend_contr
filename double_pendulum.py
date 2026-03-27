#!/usr/bin/env python3
"""
Double Pendulum Simulation — Full Nonlinear Lagrangian Dynamics
===============================================================

Physically accurate simulation of a double pendulum using:
  - Full nonlinear equations of motion (Lagrangian formalism)
  - DOP853 integrator (8th-order Runge-Kutta) with tight tolerances
  - Real-time matplotlib visualization with persistent trails
  - Live energy monitoring as an integration accuracy check

Dependencies: numpy, scipy, matplotlib (all standard scientific Python).

Author: Claude (Anthropic)
"""

# =============================================================================
# ANGLE CONVENTION
# =============================================================================
#
# Both theta_1 and theta_2 are measured from the DOWNWARD VERTICAL (global frame).
#
#         pivot
#           |      <- 180° (up, unstable equilibrium)
#           |
#     ------+------  <- 90° (horizontal)
#           |
#           |      <- 0° (down, stable equilibrium / resting position)
#           o  bob
#
# theta_1: angle of rod 1 from downward vertical (absolute)
# theta_2: angle of rod 2 from downward vertical (absolute, NOT relative to rod 1)
#
# Positive angles are measured counterclockwise from the downward vertical.
#
# The equations of motion below use this same convention directly.
# The angle that enters sin/cos in the EOM is measured from the downward
# vertical, so the user-facing convention maps 1:1 to the internal
# representation — no conversion is needed. We make this explicit in the code.
#
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================================================================
# Physical parameters
# =============================================================================
L1 = 1.0      # length of rod 1 [m]
L2 = 1.0      # length of rod 2 [m]
M1 = 1.0      # mass of bob 1 [kg]
M2 = 1.0      # mass of bob 2 [kg]
G  = 9.81     # gravitational acceleration [m/s^2]

# =============================================================================
# Visualization parameters
# =============================================================================
TRAIL_LENGTH = 2000   # number of trail points to keep per bob
DT_FRAME     = 0.02   # time step per animation frame [s] (50 fps target)
CHUNK_TIME   = 0.5    # integration chunk duration [s] per solve_ivp call


# =============================================================================
# Equations of Motion — Lagrangian Formalism
# =============================================================================
#
# We model two point masses m1, m2 connected by rigid massless rods of
# lengths L1, L2 to a fixed pivot.
#
# Generalized coordinates: theta_1, theta_2  (angles from downward vertical)
#
# Positions (pivot at origin, y-axis pointing UP):
#   x1 =  L1 * sin(theta_1)
#   y1 = -L1 * cos(theta_1)
#
#   x2 =  L1 * sin(theta_1) + L2 * sin(theta_2)
#   y2 = -L1 * cos(theta_1) - L2 * cos(theta_2)
#
# Velocities:
#   dx1/dt =  L1 * cos(theta_1) * omega_1
#   dy1/dt =  L1 * sin(theta_1) * omega_1
#
#   dx2/dt =  L1 * cos(theta_1) * omega_1 + L2 * cos(theta_2) * omega_2
#   dy2/dt =  L1 * sin(theta_1) * omega_1 + L2 * sin(theta_2) * omega_2
#
# Kinetic energy:
#   T = 0.5 * m1 * (dx1² + dy1²) + 0.5 * m2 * (dx2² + dy2²)
#     = 0.5 * (m1 + m2) * L1² * omega_1²
#       + 0.5 * m2 * L2² * omega_2²
#       + m2 * L1 * L2 * cos(theta_1 - theta_2) * omega_1 * omega_2
#
# Potential energy (reference: pivot height):
#   V = m1 * g * y1 + m2 * g * y2
#     = -(m1 + m2) * g * L1 * cos(theta_1) - m2 * g * L2 * cos(theta_2)
#
# Lagrangian: L = T - V
#
# Applying the Euler-Lagrange equations:
#   d/dt (dL/d(omega_i)) - dL/d(theta_i) = 0,   i = 1, 2
#
# yields the system:  M * [alpha_1, alpha_2]^T = f
#
# where alpha_i = d(omega_i)/dt  are the angular accelerations,
# and delta = theta_1 - theta_2.
#
# ---- Mass matrix M (2x2, symmetric) ----
#
# M[0,0] = (m1 + m2) * L1²
#   -- inertia of the full system about the pivot
#
# M[0,1] = m2 * L1 * L2 * cos(delta)
#   -- coupling between the two rods (off-diagonal)
#
# M[1,0] = m2 * L1 * L2 * cos(delta)
#   -- symmetric with M[0,1]
#
# M[1,1] = m2 * L2²
#   -- inertia of mass 2 about the elbow joint
#
# ---- Forcing vector f (2x1) ----
#
# f[0] = -m2 * L1 * L2 * sin(delta) * omega_2²
#         - (m1 + m2) * g * L1 * sin(theta_1)
#   -- centripetal coupling from rod 2 + gravitational torque on rod 1
#
# f[1] = +m2 * L1 * L2 * sin(delta) * omega_1²
#         - m2 * g * L2 * sin(theta_2)
#   -- centripetal coupling from rod 1 + gravitational torque on rod 2
#
# ---- Analytic inverse of the 2x2 mass matrix ----
#
# For a 2x2 matrix  M = [[a, b], [c, d]]:
#   det(M) = a*d - b*c
#   M^{-1} = (1/det) * [[d, -b], [-c, a]]
#
# Here:
#   det(M) = (m1 + m2) * L1² * m2 * L2²  -  (m2 * L1 * L2 * cos(delta))²
#          = m2 * L1² * L2² * [(m1 + m2) - m2 * cos²(delta)]
#          = m2 * L1² * L2² * [m1 + m2 * sin²(delta)]
#
# This determinant is always positive (m1 > 0), so M is always invertible.
#
# M^{-1}[0,0] =  m2 * L2²                    / det
# M^{-1}[0,1] = -m2 * L1 * L2 * cos(delta)   / det
# M^{-1}[1,0] = -m2 * L1 * L2 * cos(delta)   / det
# M^{-1}[1,1] =  (m1 + m2) * L1²             / det
#
# Angular accelerations:
#   [alpha_1, alpha_2]^T = M^{-1} * f
# =============================================================================


def derivatives(t, state):
    """
    Compute the time derivatives of the state vector.

    Parameters
    ----------
    t : float
        Current time [s] (unused, system is autonomous).
    state : array-like, shape (4,)
        [theta_1, omega_1, theta_2, omega_2]
        theta_i in radians (from downward vertical), omega_i in rad/s.

    Returns
    -------
    dstate : ndarray, shape (4,)
        [omega_1, alpha_1, omega_2, alpha_2]
    """
    theta1, omega1, theta2, omega2 = state

    # Angle difference
    delta = theta1 - theta2       # [rad]
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)

    # ------------------------------------------------------------------
    # Mass matrix entries
    # ------------------------------------------------------------------
    # M[0,0] = (m1 + m2) * L1^2
    M00 = (M1 + M2) * L1 * L1

    # M[0,1] = M[1,0] = m2 * L1 * L2 * cos(delta)
    M01 = M2 * L1 * L2 * cos_d

    # M[1,1] = m2 * L2^2
    M11 = M2 * L2 * L2

    # ------------------------------------------------------------------
    # Forcing vector entries
    # ------------------------------------------------------------------
    # f[0] = -m2 * L1 * L2 * sin(delta) * omega_2^2
    #        - (m1 + m2) * g * L1 * sin(theta_1)
    f0 = (-M2 * L1 * L2 * sin_d * omega2 * omega2
           - (M1 + M2) * G * L1 * np.sin(theta1))

    # f[1] = +m2 * L1 * L2 * sin(delta) * omega_1^2
    #        - m2 * g * L2 * sin(theta_2)
    f1 = (M2 * L1 * L2 * sin_d * omega1 * omega1
          - M2 * G * L2 * np.sin(theta2))

    # ------------------------------------------------------------------
    # Determinant of M
    # ------------------------------------------------------------------
    # det = M[0,0] * M[1,1] - M[0,1]^2
    #     = m2 * L1^2 * L2^2 * (m1 + m2 * sin^2(delta))
    det = M00 * M11 - M01 * M01

    # ------------------------------------------------------------------
    # Angular accelerations via analytic inverse:
    #   alpha = M^{-1} * f
    #
    #   alpha_1 = ( M[1,1] * f[0] - M[0,1] * f[1]) / det
    #   alpha_2 = (-M[1,0] * f[0] + M[0,0] * f[1]) / det
    # ------------------------------------------------------------------
    alpha1 = (M11 * f0 - M01 * f1) / det
    alpha2 = (-M01 * f0 + M00 * f1) / det

    return np.array([omega1, alpha1, omega2, alpha2])


def compute_energy(state):
    """
    Compute kinetic, potential, and total mechanical energy.

    Parameters
    ----------
    state : array-like, shape (4,)
        [theta_1, omega_1, theta_2, omega_2]

    Returns
    -------
    T : float   — kinetic energy [J]
    V : float   — potential energy [J]
    E : float   — total energy [J]
    """
    theta1, omega1, theta2, omega2 = state

    # Kinetic energy [J]:
    # T = 0.5*(m1+m2)*L1^2*omega_1^2
    #   + 0.5*m2*L2^2*omega_2^2
    #   + m2*L1*L2*cos(theta_1 - theta_2)*omega_1*omega_2
    T = (0.5 * (M1 + M2) * L1 * L1 * omega1 * omega1
         + 0.5 * M2 * L2 * L2 * omega2 * omega2
         + M2 * L1 * L2 * np.cos(theta1 - theta2) * omega1 * omega2)

    # Potential energy [J] (reference: pivot height, y-axis up):
    # V = -(m1+m2)*g*L1*cos(theta_1) - m2*g*L2*cos(theta_2)
    V = (-(M1 + M2) * G * L1 * np.cos(theta1)
         - M2 * G * L2 * np.cos(theta2))

    return T, V, T + V


def positions_from_state(state):
    """
    Compute Cartesian positions of both bobs from the state vector.

    Parameters
    ----------
    state : array-like, shape (4,)
        [theta_1, omega_1, theta_2, omega_2]

    Returns
    -------
    x1, y1, x2, y2 : float
        Cartesian coordinates [m] of bob 1 and bob 2.
        Pivot is at origin; y-axis points UP.
    """
    theta1, omega1, theta2, omega2 = state

    # Bob 1 position
    x1 =  L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    # Bob 2 position
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2


# =============================================================================
# Interactive CLI for initial conditions
# =============================================================================

def get_initial_conditions():
    """
    Prompt the user for initial conditions via a CLI interface.

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
            # Classic chaotic preset
            theta1_deg = 120.0
            theta2_deg = -20.0
            omega1_deg = 0.0
            omega2_deg = 0.0
            print()
            print(f"  Using preset: θ₁={theta1_deg}°, θ₂={theta2_deg}°, "
                  f"ω₁={omega1_deg}°/s, ω₂={omega2_deg}°/s")
        else:
            theta1_deg = float(raw)
            theta2_deg = float(input("  θ₂ [degrees, default=-20]: ").strip() or "-20")
            omega1_deg = float(input("  ω₁ [degrees/s, default=0]: ").strip() or "0")
            omega2_deg = float(input("  ω₂ [degrees/s, default=0]: ").strip() or "0")
    except (ValueError, EOFError):
        print("  Invalid input — using preset.")
        theta1_deg = 120.0
        theta2_deg = -20.0
        omega1_deg = 0.0
        omega2_deg = 0.0

    # ---------------------------------------------------------------
    # Convert degrees → radians.
    #
    # ANGLE CONVENTION NOTE:
    # The user enters angles in the "from downward vertical" convention.
    # The equations of motion use the same convention, so this is a
    # direct conversion with no additional offset — intentionally a
    # 1:1 mapping, not an oversight.
    # ---------------------------------------------------------------
    theta1 = np.radians(theta1_deg)   # [deg] → [rad], no offset needed
    theta2 = np.radians(theta2_deg)   # [deg] → [rad], no offset needed
    omega1 = np.radians(omega1_deg)   # [deg/s] → [rad/s]
    omega2 = np.radians(omega2_deg)   # [deg/s] → [rad/s]

    state0 = np.array([theta1, omega1, theta2, omega2])

    print()
    print(f"  Internal state: θ₁={theta1:.4f} rad, ω₁={omega1:.4f} rad/s")
    print(f"                  θ₂={theta2:.4f} rad, ω₂={omega2:.4f} rad/s")
    print()

    return state0


# =============================================================================
# Simulation + Visualization
# =============================================================================

def run_simulation(state0):
    """
    Run the double pendulum simulation with real-time visualization.

    Integration is performed live in chunks using DOP853 (8th-order
    Runge-Kutta) with tight tolerances.  The animation runs until the
    user closes the matplotlib window.

    Parameters
    ----------
    state0 : ndarray, shape (4,)
        Initial state [theta_1, omega_1, theta_2, omega_2].
    """
    # ----- Simulation state (mutable, updated by integration loop) -----
    sim = {
        "state": state0.copy(),
        "time": 0.0,
        "buffer_t": [],      # time values from latest integration chunk
        "buffer_y": [],      # state vectors from latest integration chunk
        "buf_idx": 0,        # current read index into the buffer
    }

    # Initial energy — used for drift monitoring
    T0, V0, E0 = compute_energy(state0)
    sim["E0"] = E0

    # Trail storage (deques auto-discard old points)
    trail1_x = deque(maxlen=TRAIL_LENGTH)
    trail1_y = deque(maxlen=TRAIL_LENGTH)
    trail2_x = deque(maxlen=TRAIL_LENGTH)
    trail2_y = deque(maxlen=TRAIL_LENGTH)

    energy_warned = [False]  # mutable flag for the 0.1% warning

    # ----- Pre-fill the first integration chunk -----
    def integrate_chunk():
        """Integrate forward by CHUNK_TIME seconds and store results."""
        t_start = sim["time"]
        t_end = t_start + CHUNK_TIME

        # Number of evaluation points in this chunk
        n_points = max(int(CHUNK_TIME / DT_FRAME), 2)
        t_eval = np.linspace(t_start, t_end, n_points + 1)

        sol = solve_ivp(
            fun=derivatives,
            t_span=(t_start, t_end),
            y0=sim["state"],
            method="DOP853",        # 8th-order Runge-Kutta
            t_eval=t_eval,
            rtol=1e-10,             # relative tolerance
            atol=1e-12,             # absolute tolerance
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed at t={t_start:.3f}s: {sol.message}")

        # Store the chunk (skip the first point — it duplicates the end of the
        # previous chunk, except for the very first call).
        start = 1 if sim["time"] > 0 else 0
        sim["buffer_t"] = sol.t[start:].tolist()
        sim["buffer_y"] = sol.y[:, start:].T.tolist()  # shape (n, 4)
        sim["buf_idx"] = 0

        # Advance state to end of chunk for next call
        sim["state"] = sol.y[:, -1].copy()
        sim["time"] = sol.t[-1]

    integrate_chunk()  # fill the first chunk

    # ----- Set up the matplotlib figure -----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    limit = (L1 + L2) * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Double Pendulum — Full Nonlinear Simulation")
    ax.grid(True, alpha=0.3)

    # Pivot marker
    ax.plot(0, 0, "ko", markersize=6, zorder=5)

    # Pendulum rods
    rod_line, = ax.plot([], [], "k-", linewidth=2, zorder=4)

    # Bob markers
    bob1, = ax.plot([], [], "o", color="royalblue", markersize=12,
                    markeredgecolor="black", zorder=5)
    bob2, = ax.plot([], [], "o", color="crimson", markersize=12,
                    markeredgecolor="black", zorder=5)

    # Trail lines
    trail1_line, = ax.plot([], [], "-", color="royalblue", linewidth=0.8,
                           alpha=0.6, zorder=2)
    trail2_line, = ax.plot([], [], "-", color="crimson", linewidth=0.8,
                           alpha=0.6, zorder=2)

    # Text overlays
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        fontsize=10, verticalalignment="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="wheat", alpha=0.8))
    energy_text = ax.text(0.02, 0.82, "", transform=ax.transAxes,
                          fontsize=9, verticalalignment="top",
                          fontfamily="monospace",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="lightyellow", alpha=0.8))

    def init():
        """Initialize animation artists."""
        rod_line.set_data([], [])
        bob1.set_data([], [])
        bob2.set_data([], [])
        trail1_line.set_data([], [])
        trail2_line.set_data([], [])
        time_text.set_text("")
        energy_text.set_text("")
        return rod_line, bob1, bob2, trail1_line, trail2_line, time_text, energy_text

    def update(frame):
        """Advance one frame: read from buffer, refill if needed."""
        # Fetch next state from the integration buffer
        if sim["buf_idx"] >= len(sim["buffer_t"]):
            integrate_chunk()

        idx = sim["buf_idx"]
        t_now = sim["buffer_t"][idx]
        state_now = np.array(sim["buffer_y"][idx])
        sim["buf_idx"] = idx + 1

        # Positions
        x1, y1, x2, y2 = positions_from_state(state_now)

        # Update rods: pivot → bob1 → bob2
        rod_line.set_data([0, x1, x2], [0, y1, y2])

        # Update bob markers
        bob1.set_data([x1], [y1])
        bob2.set_data([x2], [y2])

        # Append to trails
        trail1_x.append(x1)
        trail1_y.append(y1)
        trail2_x.append(x2)
        trail2_y.append(y2)
        trail1_line.set_data(list(trail1_x), list(trail1_y))
        trail2_line.set_data(list(trail2_x), list(trail2_y))

        # Energy readout
        T, V, E = compute_energy(state_now)
        drift_pct = abs((E - sim["E0"]) / sim["E0"]) * 100.0 if sim["E0"] != 0 else 0.0

        # Warn once if energy drifts > 0.1%
        if drift_pct > 0.1 and not energy_warned[0]:
            energy_warned[0] = True
            print(f"  ⚠ WARNING: Total energy drift exceeded 0.1% at t={t_now:.2f}s "
                  f"(drift={drift_pct:.4f}%)")

        # Time overlay
        time_text.set_text(f"t = {t_now:8.3f} s")

        # Energy overlay
        energy_text.set_text(
            f"KE  = {T:+10.5f} J\n"
            f"PE  = {V:+10.5f} J\n"
            f"Tot = {E:+10.5f} J\n"
            f"ΔE  = {drift_pct:8.6f} %"
        )

        return rod_line, bob1, bob2, trail1_line, trail2_line, time_text, energy_text

    # Run the animation indefinitely (until window is closed)
    anim = FuncAnimation(
        fig, update, init_func=init,
        interval=DT_FRAME * 1000,  # milliseconds between frames
        blit=True,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Entry point: get initial conditions, then run the simulation."""
    state0 = get_initial_conditions()

    print("=" * 60)
    print("  Starting simulation...")
    print("  Close the matplotlib window to stop.")
    print("=" * 60)
    print()

    _, _, E0 = compute_energy(state0)
    print(f"  Initial total energy: {E0:.6f} J")
    print()

    run_simulation(state0)

    print("  Simulation ended.")


if __name__ == "__main__":
    main()
