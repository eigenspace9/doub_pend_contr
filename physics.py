"""
Double Pendulum Physics Engine
==============================

Full nonlinear equations of motion derived from Lagrangian mechanics.
All dynamics are expressed in explicit state-space matrix form:

    dx/dt = g(x)

where x = [theta_1, omega_1, theta_2, omega_2]^T  is the state vector.

The angular accelerations are computed via:

    M(theta) * [alpha_1, alpha_2]^T = f(theta, omega)
    [alpha_1, alpha_2]^T = M^{-1}(theta) * f(theta, omega)

Everything is built with explicit numpy matrices — no compressed one-liners.

ANGLE CONVENTION:
    Both theta_1 and theta_2 are measured from the DOWNWARD VERTICAL.
    0 deg   = hanging straight down  (stable equilibrium)
    180 deg = pointing straight up   (unstable equilibrium)
    90 deg  = horizontal
    theta_2 is ABSOLUTE, not relative to rod 1.
"""

import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# Physical parameters
# =============================================================================
L1 = 1.0      # length of rod 1 [m]
L2 = 1.0      # length of rod 2 [m]
M1 = 1.0      # mass of bob 1 [kg]
M2 = 1.0      # mass of bob 2 [kg]
G  = 9.81     # gravitational acceleration [m/s^2]

# =============================================================================
# Integration parameters
# =============================================================================
CHUNK_TIME = 0.5    # integration chunk duration [s] per solve_ivp call
DT_FRAME   = 0.02   # requested output spacing [s] (~50 fps)


# =============================================================================
# State-Space Model
# =============================================================================
#
# STATE VECTOR (4 x 1):
#
#         [ theta_1 ]        angle of rod 1 from downward vertical  [rad]
#   x  =  [ omega_1 ]        angular velocity of rod 1              [rad/s]
#         [ theta_2 ]        angle of rod 2 from downward vertical  [rad]
#         [ omega_2 ]        angular velocity of rod 2              [rad/s]
#
#
# STATE EQUATION:
#
#   dx/dt = g(x)
#
#         [ omega_1                    ]
#   g(x)= [ (M^{-1} * f)[0]           ]       ... angular acceleration of rod 1
#         [ omega_2                    ]
#         [ (M^{-1} * f)[1]           ]       ... angular acceleration of rod 2
#
#
# Or equivalently, splitting into kinematic + dynamic parts:
#
#   [ d(theta_1)/dt ]   [ 0  1  0  0 ] [ theta_1 ]   [       0       ]
#   [ d(omega_1)/dt ] = [ 0  0  0  0 ] [ omega_1 ] + [ (M^-1 * f)[0] ]
#   [ d(theta_2)/dt ]   [ 0  0  0  1 ] [ theta_2 ]   [       0       ]
#   [ d(omega_2)/dt ]   [ 0  0  0  0 ] [ omega_2 ]   [ (M^-1 * f)[1] ]
#
#   ^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^  ^^^^^^^^^^    ^^^^^^^^^^^^^^^^^
#       dx/dt          kinematic matrix    x          nonlinear dynamics
#                       (trivial part)                  (from Lagrangian)
#
# The first matrix just says "angle derivative = velocity".
# All the interesting physics lives in M^{-1} * f.
#
#
# MASS MATRIX M (2 x 2, symmetric, positive-definite):
#
#        [ (m1 + m2) * L1^2             m2 * L1 * L2 * cos(delta) ]
#   M =  [                                                         ]
#        [ m2 * L1 * L2 * cos(delta)    m2 * L2^2                 ]
#
#   where delta = theta_1 - theta_2
#
#   M[0,0] = (m1 + m2) * L1^2           total inertia about the pivot
#   M[0,1] = m2 * L1 * L2 * cos(delta)  coupling between the two rods
#   M[1,0] = m2 * L1 * L2 * cos(delta)  (symmetric)
#   M[1,1] = m2 * L2^2                  inertia of bob 2 about the elbow
#
#
# FORCING VECTOR f (2 x 1):
#
#        [ -m2 * L1 * L2 * sin(delta) * omega_2^2  -  (m1 + m2) * g * L1 * sin(theta_1) ]
#   f =  [                                                                                 ]
#        [ +m2 * L1 * L2 * sin(delta) * omega_1^2  -  m2 * g * L2 * sin(theta_2)         ]
#
#   f[0]: centripetal torque from rod 2 swinging + gravity torque on rod 1
#   f[1]: centripetal torque from rod 1 swinging + gravity torque on rod 2
#
#
# ANALYTIC INVERSE OF M (2 x 2):
#
#   For M = [[a, b], [c, d]]:
#       det(M) = a*d - b*c
#       M^{-1} = (1/det) * [[ d, -b],
#                            [-c,  a]]
#
#   det(M) = (m1+m2)*L1^2 * m2*L2^2  -  (m2*L1*L2*cos(delta))^2
#          = m2 * L1^2 * L2^2 * [ m1 + m2*sin^2(delta) ]
#
#   Always positive (m1 > 0), so M is always invertible.
#
#   M^{-1} = (1/det) * [[ m2*L2^2,                  -m2*L1*L2*cos(delta) ],
#                        [ -m2*L1*L2*cos(delta),      (m1+m2)*L1^2       ]]
#
#
# ANGULAR ACCELERATIONS:
#
#   [ alpha_1 ]           [ f[0] ]
#   [         ] = M^{-1} *[      ]
#   [ alpha_2 ]           [ f[1] ]
#
# =============================================================================


def build_mass_matrix(theta1, theta2):
    """
    Build the 2x2 mass (inertia) matrix M for the double pendulum.

    Parameters
    ----------
    theta1, theta2 : float
        Angles from downward vertical [rad].

    Returns
    -------
    M : ndarray, shape (2, 2)
        Symmetric positive-definite mass matrix.
    """
    delta = theta1 - theta2   # angle difference [rad]
    cos_d = np.cos(delta)

    M = np.array([
        # Row 0:
        [
            (M1 + M2) * L1 * L1,          # M[0,0] = (m1+m2)*L1^2
            M2 * L1 * L2 * cos_d,          # M[0,1] = m2*L1*L2*cos(delta)
        ],
        # Row 1:
        [
            M2 * L1 * L2 * cos_d,          # M[1,0] = m2*L1*L2*cos(delta)
            M2 * L2 * L2,                   # M[1,1] = m2*L2^2
        ],
    ])

    return M


def build_forcing_vector(theta1, omega1, theta2, omega2):
    """
    Build the 2x1 forcing (right-hand side) vector f.

    Parameters
    ----------
    theta1, omega1 : float   — angle [rad] and angular velocity [rad/s] of rod 1
    theta2, omega2 : float   — angle [rad] and angular velocity [rad/s] of rod 2

    Returns
    -------
    f : ndarray, shape (2,)
        Generalized forces (centripetal + gravitational torques).
    """
    delta = theta1 - theta2   # angle difference [rad]
    sin_d = np.sin(delta)

    f = np.array([
        # f[0] = -m2*L1*L2*sin(delta)*omega_2^2 - (m1+m2)*g*L1*sin(theta_1)
        #         centripetal from rod 2            gravity on rod 1
        -M2 * L1 * L2 * sin_d * omega2**2  -  (M1 + M2) * G * L1 * np.sin(theta1),

        # f[1] = +m2*L1*L2*sin(delta)*omega_1^2 - m2*g*L2*sin(theta_2)
        #         centripetal from rod 1            gravity on rod 2
        +M2 * L1 * L2 * sin_d * omega1**2  -  M2 * G * L2 * np.sin(theta2),
    ])

    return f


def invert_mass_matrix(M_mat):
    """
    Analytically invert the 2x2 mass matrix.

    For M = [[a, b], [c, d]]:
        det = a*d - b*c
        M^{-1} = (1/det) * [[ d, -b], [-c, a]]

    Parameters
    ----------
    M_mat : ndarray, shape (2, 2)

    Returns
    -------
    M_inv : ndarray, shape (2, 2)
    det   : float   — determinant (useful for diagnostics)
    """
    a, b = M_mat[0, 0], M_mat[0, 1]
    c, d = M_mat[1, 0], M_mat[1, 1]

    # det = (m1+m2)*L1^2 * m2*L2^2  -  (m2*L1*L2*cos(delta))^2
    #     = m2*L1^2*L2^2 * (m1 + m2*sin^2(delta))     > 0 always
    det = a * d - b * c

    M_inv = (1.0 / det) * np.array([
        [ d, -b],       # M_inv[0,0] =  m2*L2^2 / det
                        # M_inv[0,1] = -m2*L1*L2*cos(delta) / det
        [-c,  a],       # M_inv[1,0] = -m2*L1*L2*cos(delta) / det
                        # M_inv[1,1] =  (m1+m2)*L1^2 / det
    ])

    return M_inv, det


def derivatives(t, state, torque=None):
    """
    Right-hand side of the state-space equation: dx/dt = g(x).

    Computes angular accelerations by solving  M * alpha = f + tau
    via the analytic inverse of M.

    Parameters
    ----------
    t : float
        Current time [s] (unused — autonomous system).
    state : array-like, shape (4,)
        [theta_1, omega_1, theta_2, omega_2]
    torque : array-like, shape (2,), optional
        External torques [tau_1, tau_2] applied to the two joints [N*m].
        tau_1 acts at the pivot (on rod 1), tau_2 at the elbow (on rod 2).
        Default: no external torques.

    Returns
    -------
    dstate : ndarray, shape (4,)
        [d(theta_1)/dt, d(omega_1)/dt, d(theta_2)/dt, d(omega_2)/dt]
      = [omega_1,       alpha_1,       omega_2,       alpha_2      ]
    """
    theta1, omega1, theta2, omega2 = state

    # --- Build the 2x2 mass matrix ---
    M_mat = build_mass_matrix(theta1, theta2)

    # --- Build the 2x1 forcing vector ---
    f = build_forcing_vector(theta1, omega1, theta2, omega2)

    # --- Add external torques if provided ---
    if torque is not None:
        # tau is added directly to the generalized force vector
        # tau[0] acts on joint 1 (pivot), tau[1] acts on joint 2 (elbow)
        f = f + np.asarray(torque)

    # --- Invert mass matrix analytically ---
    M_inv, det = invert_mass_matrix(M_mat)

    # --- Compute angular accelerations: alpha = M^{-1} * f ---
    #
    #   [ alpha_1 ]   [ M_inv[0,0]  M_inv[0,1] ]   [ f[0] ]
    #   [         ] = [                          ] * [      ]
    #   [ alpha_2 ]   [ M_inv[1,0]  M_inv[1,1] ]   [ f[1] ]
    #
    alpha = M_inv @ f     # matrix-vector product, shape (2,)

    # --- Assemble full state derivative ---
    #
    #   dx/dt = [ omega_1,  alpha_1,  omega_2,  alpha_2 ]^T
    #
    dstate = np.array([
        omega1,       # d(theta_1)/dt = omega_1          [rad/s]
        alpha[0],     # d(omega_1)/dt = alpha_1           [rad/s^2]
        omega2,       # d(theta_2)/dt = omega_2          [rad/s]
        alpha[1],     # d(omega_2)/dt = alpha_2           [rad/s^2]
    ])

    return dstate


# =============================================================================
# Energy computation
# =============================================================================

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
    E : float   — total energy T + V [J]
    """
    theta1, omega1, theta2, omega2 = state

    # Kinetic energy [J]:
    #   T = 0.5*(m1+m2)*L1^2*omega_1^2
    #     + 0.5*m2*L2^2*omega_2^2
    #     + m2*L1*L2*cos(theta_1 - theta_2)*omega_1*omega_2
    T = (0.5 * (M1 + M2) * L1**2 * omega1**2
         + 0.5 * M2 * L2**2 * omega2**2
         + M2 * L1 * L2 * np.cos(theta1 - theta2) * omega1 * omega2)

    # Potential energy [J] (reference: pivot height, y pointing up):
    #   V = -(m1+m2)*g*L1*cos(theta_1) - m2*g*L2*cos(theta_2)
    V = (-(M1 + M2) * G * L1 * np.cos(theta1)
         - M2 * G * L2 * np.cos(theta2))

    return T, V, T + V


# =============================================================================
# Cartesian positions
# =============================================================================

def positions_from_state(state):
    """
    Convert state vector to Cartesian positions of both bobs.

    Pivot at origin, y-axis points UP.

    Returns
    -------
    x1, y1, x2, y2 : float   — positions [m]
    """
    theta1, omega1, theta2, omega2 = state

    # Bob 1
    x1 =  L1 * np.sin(theta1)          # [m]
    y1 = -L1 * np.cos(theta1)          # [m]

    # Bob 2 (absolute position, not relative)
    x2 = x1 + L2 * np.sin(theta2)      # [m]
    y2 = y1 - L2 * np.cos(theta2)      # [m]

    return x1, y1, x2, y2


# =============================================================================
# Integration engine
# =============================================================================

def integrate_chunk(state, t_start, chunk_time=CHUNK_TIME, dt=DT_FRAME):
    """
    Integrate the equations of motion forward by one time chunk.

    Uses DOP853 (8th-order Runge-Kutta, Dormand-Prince) with tight
    tolerances for maximum accuracy.

    Parameters
    ----------
    state : ndarray, shape (4,)
        Current state at t_start.
    t_start : float
        Start time [s].
    chunk_time : float
        Duration to integrate [s].
    dt : float
        Approximate output spacing [s].

    Returns
    -------
    t_array : ndarray   — evaluation times [s]
    y_array : ndarray, shape (n, 4)   — state at each time
    """
    t_end = t_start + chunk_time
    n_points = max(int(chunk_time / dt), 2)
    t_eval = np.linspace(t_start, t_end, n_points + 1)

    sol = solve_ivp(
        fun=derivatives,
        t_span=(t_start, t_end),
        y0=state,
        method="DOP853",        # 8th-order Runge-Kutta (Dormand-Prince 8(5,3))
        t_eval=t_eval,
        rtol=1e-10,             # relative tolerance
        atol=1e-12,             # absolute tolerance
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed at t={t_start:.3f}s: {sol.message}")

    return sol.t, sol.y.T     # y.T has shape (n_points, 4)


# =============================================================================
# Linearized state-space model around the upright equilibrium
# =============================================================================
#
# For CONTROLLER DESIGN we linearize about the unstable equilibrium:
#   theta_1 = pi,  theta_2 = pi,  omega_1 = omega_2 = 0
#
# Define perturbation variables:
#   phi_1 = theta_1 - pi       (small deviation from upright)
#   phi_2 = theta_2 - pi
#
# Linearization of trig functions:
#   sin(theta_i) = sin(pi + phi_i) = -sin(phi_i) ≈ -phi_i
#   cos(delta)   = cos(phi_1 - phi_2)             ≈ 1
#   sin(delta)   = sin(phi_1 - phi_2)             ≈ phi_1 - phi_2
#   omega_i^2    ≈ 0   (second-order small)
#
# Mass matrix at equilibrium (delta = 0, cos(delta) = 1):
#
#           [ (m1+m2)*L1^2     m2*L1*L2 ]
#   M_0  =  [                            ]
#           [ m2*L1*L2         m2*L2^2   ]
#
# Linearized forcing (dropping omega^2 terms):
#
#           [ +(m1+m2)*g*L1*phi_1 ]       gravity is now DESTABILIZING
#   f_lin = [                      ]       (positive feedback — pushes away
#           [ +m2*g*L2*phi_2      ]        from upright)
#
# With a control torque tau applied at the pivot:
#
#           [ +(m1+m2)*g*L1*phi_1 + tau ]
#   f_lin = [                            ]
#           [ +m2*g*L2*phi_2            ]
#
# The linearized state-space model is:
#
#   d/dt [phi_1  ]   [0   1   0   0] [phi_1  ]   [0]
#        [dphi_1 ] = [A21 0  A23  0] [dphi_1 ] + [B2] * tau
#        [phi_2  ]   [0   0   0   1] [phi_2  ]   [0]
#        [dphi_2 ]   [A41 0  A43  0] [dphi_2 ]   [B4]
#
#   where the A and B entries come from M_0^{-1} applied to the gravity
#   and input terms.  See compute_linearized_ss() below.
#
# =============================================================================


def compute_linearized_ss():
    """
    Compute the linearized state-space matrices (A, B) around the
    upright equilibrium (theta_1 = theta_2 = pi).

    The state is x = [phi_1, dphi_1/dt, phi_2, dphi_2/dt]^T
    where phi_i = theta_i - pi.

    Control input: u = tau  (torque at the pivot joint) [N*m].

    Returns
    -------
    A : ndarray, shape (4, 4)   — system matrix
    B : ndarray, shape (4, 1)   — input matrix
    M0 : ndarray, shape (2, 2) — mass matrix at equilibrium
    M0_inv : ndarray, shape (2, 2) — its inverse
    """
    # Mass matrix at equilibrium (cos(delta) = 1)
    #
    #   M_0 = [ (m1+m2)*L1^2,   m2*L1*L2 ]
    #         [ m2*L1*L2,        m2*L2^2  ]
    M0 = np.array([
        [(M1 + M2) * L1**2,    M2 * L1 * L2],
        [M2 * L1 * L2,         M2 * L2**2  ],
    ])

    # det(M_0) = (m1+m2)*m2*L1^2*L2^2 - m2^2*L1^2*L2^2
    #          = m1*m2*L1^2*L2^2
    det_M0 = M1 * M2 * L1**2 * L2**2

    # Analytic inverse
    M0_inv = (1.0 / det_M0) * np.array([
        [ M2 * L2**2,        -M2 * L1 * L2],
        [-M2 * L1 * L2,      (M1 + M2) * L1**2],
    ])

    # Gravity stiffness matrix K_g (maps [phi_1, phi_2] -> generalized force)
    #   K_g = diag( (m1+m2)*g*L1,  m2*g*L2 )
    # Note: positive sign = destabilizing (unstable equilibrium)
    K_g = np.array([
        [(M1 + M2) * G * L1,    0.0          ],
        [0.0,                    M2 * G * L2  ],
    ])

    # M_0^{-1} * K_g  gives the [alpha_1, alpha_2] contribution from [phi_1, phi_2]
    MiK = M0_inv @ K_g     # shape (2, 2)

    # Input vector for torque at pivot: tau enters f[0] only
    #   f_tau = [tau, 0]^T
    #   M_0^{-1} * f_tau = M_0^{-1} * [1, 0]^T * tau
    B_alpha = M0_inv @ np.array([1.0, 0.0])    # shape (2,)

    # Assemble the 4x4 A matrix
    #
    #   A = [ 0,      1,      0,      0     ]
    #       [ MiK[0,0], 0,   MiK[0,1], 0    ]
    #       [ 0,      0,      0,      1     ]
    #       [ MiK[1,0], 0,   MiK[1,1], 0    ]
    A = np.array([
        [0.0,       1.0,       0.0,       0.0],
        [MiK[0, 0], 0.0,       MiK[0, 1], 0.0],
        [0.0,       0.0,       0.0,       1.0],
        [MiK[1, 0], 0.0,       MiK[1, 1], 0.0],
    ])

    # Assemble the 4x1 B matrix
    #
    #   B = [ 0,           ]
    #       [ B_alpha[0],  ]     (effect of tau on alpha_1)
    #       [ 0,           ]
    #       [ B_alpha[1],  ]     (effect of tau on alpha_2)
    B = np.array([
        [0.0],
        [B_alpha[0]],
        [0.0],
        [B_alpha[1]],
    ])

    return A, B, M0, M0_inv
