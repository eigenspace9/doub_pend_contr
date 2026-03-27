# Controller Preparation: Stabilizing the Double Pendulum at the Upright Position

## Goal

Design a controller that stabilizes the double pendulum at **theta_1 = theta_2 = 180 deg** (both rods pointing straight up — the unstable equilibrium).

## What We Already Have

The physics engine (`physics.py`) provides:

1. **Nonlinear dynamics** — `derivatives(t, state, torque=None)` already accepts an optional `torque` argument `[tau_1, tau_2]` that adds external torques to the generalized force vector.

2. **Linearized state-space model** — `compute_linearized_ss()` returns the **A** and **B** matrices of the linearized system around the upright equilibrium.

3. **Explicit mass matrix and its inverse** — available via `build_mass_matrix()` and `invert_mass_matrix()`.

---

## Linearized Model Summary

Around the upright equilibrium (theta_1 = theta_2 = pi), with perturbation variables:

```
phi_1 = theta_1 - pi       (small angle from upright)
phi_2 = theta_2 - pi
```

The linearized state vector is:

```
x = [phi_1, dphi_1/dt, phi_2, dphi_2/dt]^T
```

The state-space model is:

```
dx/dt = A * x  +  B * tau
```

where `tau` is the control torque at the pivot [N*m].

### Numerical values (for default parameters m1=m2=1, L1=L2=1, g=9.81):

```
        [ 0       1       0       0     ]
A   =   [ 19.62   0      -9.81   0     ]
        [ 0       0       0       1     ]
        [-19.62   0      19.62    0     ]

        [ 0  ]
B   =   [ 1  ]
        [ 0  ]
        [-1  ]
```

### Key observations:

- The A matrix has **positive** entries on the diagonal of the "acceleration" rows (A[1,0] = +19.62, A[3,2] = +19.62). This is the hallmark of an **unstable** equilibrium: gravity pushes the pendulum away from upright (positive feedback).
- The eigenvalues of A have positive real parts → the system is **open-loop unstable**.
- B shows that a torque at the pivot accelerates rod 1 positively and rod 2 negatively (reaction through the coupling).

---

## Controllability Check

Before designing a controller, verify that the system is controllable:

```python
from physics import compute_linearized_ss
import numpy as np

A, B, _, _ = compute_linearized_ss()

# Controllability matrix: C = [B, A*B, A^2*B, A^3*B]
n = A.shape[0]
C = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])

rank = np.linalg.matrix_rank(C)
print(f"Controllability rank: {rank} (need {n} for full controllability)")
# Expected output: rank = 4 → fully controllable with pivot torque alone
```

---

## Recommended Controller Approaches

### 1. LQR (Linear Quadratic Regulator) — recommended first step

Best for: keeping the pendulum near upright once it's already close.

```
u = -K * x
```

where K is the optimal gain matrix from solving the Riccati equation for cost:

```
J = integral( x^T Q x  +  u^T R u ) dt
```

**Tuning guidelines:**
- `Q = diag([q1, q2, q3, q4])` — penalize angle errors (q1, q3) more than velocity errors (q2, q4)
- `R` — penalize control effort (larger R = less aggressive, more energy-efficient)
- Start with `Q = diag([100, 1, 100, 1])`, `R = 0.1`

Implementation in scipy:

```python
from scipy.linalg import solve_continuous_are

Q = np.diag([100.0, 1.0, 100.0, 1.0])
R = np.array([[0.1]])

# Solve the continuous-time algebraic Riccati equation
P = solve_continuous_are(A, B, Q, R)
K = (1.0 / R[0, 0]) * B.T @ P
```

### 2. Energy-based swing-up + LQR switching

The LQR only works near the upright position. To get there from the hanging position:

1. **Swing-up phase**: Use an energy-based controller that pumps energy into the system until it reaches the energy level of the upright equilibrium.
2. **Switch to LQR** when both angles are within ~30 deg of upright and velocities are small.

Swing-up energy target:
```
E_upright = (m1+m2)*g*L1 + m2*g*L2     (both rods pointing up)
E_current = compute_energy(state)[2]     (current total energy)
```

### 3. Nonlinear MPC (advanced)

Model Predictive Control using the full nonlinear model — handles the entire swing-up and stabilization in one framework but is computationally expensive.

---

## What Needs to Be Built

| File | Purpose |
|------|---------|
| `controller.py` | Controller logic (LQR gains, swing-up law, mode switching) |
| `visualization.py` | Extend to pass torque into the integration loop |
| `physics.py` | Already prepared — `derivatives()` accepts `torque` argument |

### Integration changes needed:

The `integrate_chunk()` function currently calls `derivatives(t, state)` without torque. For closed-loop control, we need to either:

1. **Use a fixed-step integrator** so we can recompute the control torque at each step (simpler, less accurate).
2. **Wrap the controller into the derivatives function** via a closure or callback, so `solve_ivp` evaluates the control law at every internal substep (more accurate, recommended).

Option 2 example:
```python
def make_controlled_derivatives(controller_fn):
    def controlled_deriv(t, state):
        torque = controller_fn(t, state)
        return derivatives(t, state, torque=torque)
    return controlled_deriv
```

---

## Actuator Model

The current model assumes ideal torque application. For realism, consider adding:
- **Torque saturation**: `tau = np.clip(tau, -tau_max, tau_max)`
- **Motor dynamics**: first-order lag `d(tau_actual)/dt = (tau_cmd - tau_actual) / tau_motor`
- **Friction**: viscous damping `tau_friction = -b * omega`

These can be added incrementally without changing the core physics.
