"""
LQR Controller for Double Pendulum Stabilization
=================================================

Stabilizes the double pendulum at any of the four equilibrium points:
    (  0°,   0°)  — both rods hanging down     (stable, no control needed)
    (180°,   0°)  — rod 1 up, rod 2 down       (unstable)
    (  0°, 180°)  — rod 1 down, rod 2 up       (unstable)
    (180°, 180°)  — both rods pointing up      (unstable)

The controller applies a single torque at the pivot joint (joint 1).

Control law:
    u = -K * x
    x = [phi_1, dphi_1/dt, phi_2, dphi_2/dt]^T
    phi_i = theta_i - theta_i_eq           (perturbation from target)

Friction is included in the linearized model so that the LQR gain
accounts for the damping already present in the plant.
"""

import numpy as np
from scipy.linalg import solve_continuous_are

from physics import M1, M2, L1, L2, G, FRICTION_COEFF, derivatives, compute_energy

# ---------------------------------------------------------------------------
# Torque saturation [N·m]
# ---------------------------------------------------------------------------
TAU_MAX = 50.0

# ---------------------------------------------------------------------------
# LQR cost matrices  (tunable)
# ---------------------------------------------------------------------------
# Q penalizes state error: [phi1, dphi1, phi2, dphi2]
# Angle errors (indices 0, 2) are weighted 100× more than velocity errors.
Q_DEFAULT = np.diag([100.0, 1.0, 100.0, 1.0])

# R penalizes control effort.  Smaller R → more aggressive control.
R_DEFAULT = np.array([[0.05]])


# ---------------------------------------------------------------------------
# Helper: snap an angle [deg] to the nearest equilibrium (0° or 180°)
# ---------------------------------------------------------------------------

def snap_to_equilibrium(angle_deg: float) -> float:
    """Return 0.0 or 180.0, whichever is closer to angle_deg (mod 360)."""
    angle_deg = angle_deg % 360.0
    # Distance to 0° (including via 360°)  and to 180°
    d0   = min(abs(angle_deg - 0.0),   abs(angle_deg - 360.0))
    d180 = abs(angle_deg - 180.0)
    return 0.0 if d0 <= d180 else 180.0


# ---------------------------------------------------------------------------
# Linearization at an arbitrary equilibrium
# ---------------------------------------------------------------------------

def linearize_at(theta1_eq_rad: float, theta2_eq_rad: float):
    """
    Build the linearized state-space matrices (A, B) around a given
    equilibrium (theta1_eq, theta2_eq) with omega1 = omega2 = 0.

    Friction terms (-FRICTION_COEFF * omega_i) are included in A so that
    the LQR design matches the actual dissipative plant.

    State:  x = [phi_1, dphi_1/dt, phi_2, dphi_2/dt]
    Input:  u = tau_1  (torque at pivot) [N·m]

    Returns
    -------
    A : ndarray (4, 4)
    B : ndarray (4, 1)
    """
    delta_eq = theta1_eq_rad - theta2_eq_rad

    # Mass matrix at equilibrium
    M_eq = np.array([
        [(M1 + M2) * L1**2,              M2 * L1 * L2 * np.cos(delta_eq)],
        [M2 * L1 * L2 * np.cos(delta_eq), M2 * L2**2                    ],
    ])
    Mi = np.linalg.inv(M_eq)

    # Linearized gravity stiffness:
    #   d f[i] / d phi_i  =  -(mi_eff) * g * Li * cos(theta_i_eq)
    # Positive at upright (cos(pi) = -1) → destabilizing
    # Negative at hanging (cos(0) = +1)  → stabilizing
    K_g = np.diag([
        -(M1 + M2) * G * L1 * np.cos(theta1_eq_rad),
        -M2         * G * L2 * np.cos(theta2_eq_rad),
    ])

    MiK = Mi @ K_g              # gravity → acceleration coupling  (2×2)
    Mi_d = -FRICTION_COEFF * Mi # friction → acceleration coupling (2×2)

    # 4×4 system matrix
    #   rows 0,2 : kinematics  (dphi/dt = omega)
    #   rows 1,3 : dynamics    (domega/dt = MiK*phi + Mi_d*omega)
    A = np.array([
        [0.0,         1.0,         0.0,         0.0        ],
        [MiK[0, 0],   Mi_d[0, 0],  MiK[0, 1],   Mi_d[0, 1] ],
        [0.0,         0.0,         0.0,         1.0        ],
        [MiK[1, 0],   Mi_d[1, 0],  MiK[1, 1],   Mi_d[1, 1] ],
    ])

    # Input: pivot torque enters f[0] only  →  M^{-1} * [1, 0]^T
    B_alpha = Mi @ np.array([1.0, 0.0])
    B = np.array([[0.0], [B_alpha[0]], [0.0], [B_alpha[1]]])

    return A, B


# ---------------------------------------------------------------------------
# Controllability check
# ---------------------------------------------------------------------------

def controllability_rank(A, B) -> int:
    """Rank of the controllability matrix C = [B, AB, A²B, A³B]."""
    n = A.shape[0]
    C = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    return np.linalg.matrix_rank(C)


# ---------------------------------------------------------------------------
# LQR gain computation
# ---------------------------------------------------------------------------

def compute_lqr_gain(A, B, Q=None, R=None):
    """
    Solve the continuous-time algebraic Riccati equation and return K.

    u = -K * x  minimises  J = ∫(x'Qx + u'Ru) dt
    """
    if Q is None:
        Q = Q_DEFAULT
    if R is None:
        R = R_DEFAULT

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)   # K = R^{-1} B^T P  (general R)
    return K


# ---------------------------------------------------------------------------
# Controller class
# ---------------------------------------------------------------------------

class LQRController:
    """
    Full-state feedback LQR for double pendulum stabilization.

    Parameters
    ----------
    theta1_target_deg : float
        Target angle for rod 1 in degrees (will be snapped to 0° or 180°).
    theta2_target_deg : float
        Target angle for rod 2 in degrees (will be snapped to 0° or 180°).
    Q, R : ndarray, optional
        LQR cost matrices.  Defaults use the tuning guidelines from the md.

    Usage
    -----
    ctrl = LQRController(180, 180)
    torque = ctrl(t, state)   # returns [tau_1, tau_2]
    print(ctrl.last_tau)
    """

    def __init__(self, theta1_target_deg: float, theta2_target_deg: float,
                 Q=None, R=None):
        # Snap to valid equilibria
        t1 = snap_to_equilibrium(theta1_target_deg)
        t2 = snap_to_equilibrium(theta2_target_deg)
        if t1 != theta1_target_deg or t2 != theta2_target_deg:
            print(f"  [Controller] Snapped target to ({t1:.0f}°, {t2:.0f}°)")

        self.theta1_target_deg = t1
        self.theta2_target_deg = t2
        self.theta1_eq = np.radians(t1)
        self.theta2_eq = np.radians(t2)

        A, B = linearize_at(self.theta1_eq, self.theta2_eq)

        rank = controllability_rank(A, B)
        if rank < A.shape[0]:
            raise ValueError(
                f"System not controllable at ({t1}°, {t2}°): "
                f"rank = {rank} < {A.shape[0]}"
            )

        self.K = compute_lqr_gain(A, B, Q, R)
        self.last_tau = 0.0

        print(f"  [Controller] LQR gains K = {np.round(self.K, 3)}")

    def __call__(self, t: float, state) -> np.ndarray:
        """
        Compute [tau_1, tau_2] for the current state.

        Only tau_1 (pivot torque) is non-zero.
        """
        theta1, omega1, theta2, omega2 = state

        # Perturbation from target equilibrium, wrapped to [-π, π]
        phi1 = _wrap(theta1 - self.theta1_eq)
        phi2 = _wrap(theta2 - self.theta2_eq)

        x = np.array([phi1, omega1, phi2, omega2])

        tau = float(-(self.K @ x))
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)

        self.last_tau = tau
        return np.array([tau, 0.0])


def _wrap(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Factory: wrap controller into a derivatives closure for solve_ivp
# ---------------------------------------------------------------------------

def make_controlled_derivatives(controller):
    """
    Return a function  f(t, state) -> dstate  that calls physics.derivatives
    with the controller torque injected at each integrator sub-step.
    """
    def controlled_deriv(t, state):
        torque = controller(t, state)
        return derivatives(t, state, torque=torque)

    return controlled_deriv


# ---------------------------------------------------------------------------
# Energy-based swing-up controller
# ---------------------------------------------------------------------------

class SwingUpController:
    """
    Energy-pumping swing-up for the double pendulum.

    Applies torque at the pivot to drive total mechanical energy toward
    the target equilibrium energy, using a weighted velocity signal that
    accounts for both joints and a configuration-steering term near the
    target energy.
    """

    K_ENERGY = 5.0    # energy-error gain [N·m / J]
    ALPHA    = 0.3    # weight for omega2 in velocity signal
    K_STEER  = 8.0    # configuration steering gain [N·m / rad]
    VEL_THRESHOLD = 0.05  # below this, use gravity fallback [rad/s]

    def __init__(self, theta1_eq_deg: float, theta2_eq_deg: float):
        self.theta1_eq = np.radians(theta1_eq_deg)
        self.theta2_eq = np.radians(theta2_eq_deg)
        eq_state = [self.theta1_eq, 0.0, self.theta2_eq, 0.0]
        _, _, self.E_ref = compute_energy(eq_state)
        self.last_tau = 0.0
        print(f"  [SwingUp] target energy E_ref = {self.E_ref:.3f} J")

    def __call__(self, t: float, state) -> np.ndarray:
        theta1, omega1, theta2, omega2 = state
        _, _, E = compute_energy(state)

        # Weighted velocity signal (accounts for both joints)
        vel_signal = omega1 + self.ALPHA * omega2

        # Fallback when velocity is near zero: use gravity direction
        if abs(vel_signal) < self.VEL_THRESHOLD:
            grav_dir = -np.sign(np.sin(theta1))
            vel_signal = grav_dir if grav_dir != 0.0 else 1.0

        # Energy pumping
        tau = self.K_ENERGY * (self.E_ref - E) * np.sign(vel_signal)

        # Configuration steering when near target energy
        energy_err = abs(E - self.E_ref)
        energy_scale = abs(self.E_ref) + 1e-6
        if energy_err / energy_scale < 0.5:
            phi1 = _wrap(theta1 - self.theta1_eq)
            phi2 = _wrap(theta2 - self.theta2_eq)
            # Blend steering stronger as energy error shrinks
            blend = 1.0 - (energy_err / energy_scale) / 0.5
            tau += -self.K_STEER * blend * (phi1 + 0.7 * phi2)

        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        self.last_tau = tau
        return np.array([tau, 0.0])


# ---------------------------------------------------------------------------
# Hybrid controller: swing-up → LQR handoff
# ---------------------------------------------------------------------------

class HybridController:
    """
    Two-phase controller:
      1. Swing-up  — energy pumping until the pendulum enters the capture zone.
      2. LQR       — full-state feedback stabilization near the equilibrium.

    Automatically switches from swing-up to LQR when both angular errors
    fall within CAPTURE_DEG, and reverts to swing-up if LQR loses the system.
    """

    CAPTURE_DEG = 20.0   # capture / release threshold [deg]

    def __init__(self, theta1_target_deg: float, theta2_target_deg: float,
                 Q=None, R=None):
        t1 = snap_to_equilibrium(theta1_target_deg)
        t2 = snap_to_equilibrium(theta2_target_deg)

        self.theta1_target_deg = t1
        self.theta2_target_deg = t2
        self.theta1_eq = np.radians(t1)
        self.theta2_eq = np.radians(t2)

        # Detect if target is fully stable (both rods hanging down)
        self._target_is_stable = (t1 == 0.0 and t2 == 0.0)

        self._lqr = LQRController(t1, t2, Q, R)

        # No swing-up needed for stable equilibrium — LQR handles it
        if self._target_is_stable:
            self._swing = None
            self.mode = "lqr"
            print("  [Hybrid] Stable target — using LQR directly (no swing-up)")
        else:
            self._swing = SwingUpController(t1, t2)
            self.mode = "swing_up"

        self.last_tau = 0.0
        self._cap_rad = np.radians(self.CAPTURE_DEG)

    def __call__(self, t: float, state) -> np.ndarray:
        theta1, omega1, theta2, omega2 = state
        phi1 = abs(_wrap(theta1 - self.theta1_eq))
        phi2 = abs(_wrap(theta2 - self.theta2_eq))
        near = phi1 < self._cap_rad and phi2 < self._cap_rad

        if self._target_is_stable:
            # For stable target, always use LQR — it works from any angle
            torque = self._lqr(t, state)
            self.last_tau = self._lqr.last_tau
            return torque

        if near and self.mode == "swing_up":
            self.mode = "lqr"
            print("  [Hybrid] Captured — switching to LQR")
        elif not near and self.mode == "lqr":
            self.mode = "swing_up"
            print("  [Hybrid] Lost stabilization — reverting to swing-up")

        if self.mode == "lqr":
            torque = self._lqr(t, state)
            self.last_tau = self._lqr.last_tau
        else:
            torque = self._swing(t, state)
            self.last_tau = self._swing.last_tau

        return torque
