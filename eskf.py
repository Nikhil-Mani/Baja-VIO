import jax
import jax.numpy as jnp
from constants import *
from util import *
from state import *

# nonlinear nominal dynamics
def f(s: State, u: Input, dt) -> State:
    # transform acceleration into world frame
    R = quat_to_R(s.q)
    a = R @ (u.a - s.b_accel) + g
    q = quat_mul(s.q, quat_from_rotvec((u.w - s.b_gyro) * dt))

    xn = s.x + s.dx * dt + 0.5 * a * dt**2
    return State(
        x = xn,
        dx = s.dx + a * dt,
        q = q / jnp.linalg.norm(q),
        b_accel = s.b_accel,
        b_gyro = s.b_gyro,
    )

# nonlinear sensor mapping from state
def h(s: State, u_w, delta, params: CarParams, wheel_pos, slip):
    R = quat_to_R(s.q) # get orientation into R matrix
    w = u_w - s.b_gyro
    v_b = R.T @ s.dx # vel in body frame
    d_out, d_in = ackermann_angles(params, delta)
    angles = jnp.array([d_in, d_out, 0.0, 0.0])
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles), jnp.zeros(4)], axis=1) 
    v_w = v_b + jnp.cross(w, wheel_pos) # v = v + w x r
    v_long = jnp.sum(v_w * dirs, axis=1) # vel on rolling direction
    return (1.0 + slip) * v_long / params.r_w

# Jacobians from autodiff
def construct_F(s, u, dt):
    nominal = f(s, u, dt)
    def err(v):
        # [f(state + delta) - f(nominal)] * dt
        return boxminus(f(boxplus(s, ErrorState.from_vec(v)), u, dt), nominal).to_vec()

    return jax.jacfwd(err)(jnp.zeros(15))

def construct_H(s, *h_args):
    def err(v):
        return h(boxplus(s, ErrorState.from_vec(v)), *h_args)

    return jax.jacfwd(err)(jnp.zeros(15))

# Construct by perturbing the F matrix and seeing how it changes in one step
def construct_Q(s, u, dt, sig_a, sig_g, sig_ba, sig_bg):
    nominal = f(s, u, dt)
    def err(n):  # n = [accel noise (3), gyro noise (3)]
        u_n = Input(a=u.a + n[:3], w=u.w + n[3:])
        return boxminus(f(s, u_n, dt), nominal).to_vec()
    Fi = jax.jacfwd(err)(jnp.zeros(6))            # (15, 6)
    Qi = jnp.diag(jnp.concatenate([jnp.full(3, sig_a**2 / dt),
                                   jnp.full(3, sig_g**2 / dt)]))
    Q = Fi @ Qi @ Fi.T
    Q = Q.at[BA, BA].add(jnp.eye(3) * sig_ba**2 * dt)   # bias random walks
    Q = Q.at[BG, BG].add(jnp.eye(3) * sig_bg**2 * dt)
    return Q

def construct_reset_jacobian(delta_theta):
    """Map local error covariance into the corrected quaternion's error frame."""
    # boxplus applies a right rotation: q_new = q * Exp(delta_theta).
    # Its exact reset derivative is the SO(3) right Jacobian.
    th2 = delta_theta @ delta_theta
    small = th2 < 1e-8
    safe_th2 = jnp.where(small, 1.0, th2)
    th = jnp.sqrt(safe_th2)
    a = jnp.where(small, 0.5 - th2 / 24.0 + th2**2 / 720.0,
                  (1.0 - jnp.cos(th)) / safe_th2)
    b = jnp.where(small, 1.0 / 6.0 - th2 / 120.0 + th2**2 / 5040.0,
                  (th - jnp.sin(th)) / (safe_th2 * th))
    angle_skew = skew(delta_theta)
    rotation_reset = jnp.eye(3) - a * angle_skew + b * (angle_skew @ angle_skew)
    return jnp.eye(15).at[THETA, THETA].set(rotation_reset)

@jax.jit
def predict(s, P, u, dt, *noise):
    F = construct_F(s, u, dt)
    Q = construct_Q(s, u, dt, *noise)
    return f(s, u, dt), F @ P @ F.T + Q

@jax.jit
def update(s, P, z, R, *h_args):
    H = construct_H(s, *h_args)
    y = z - h(s, *h_args)
    S = H @ P @ H.T + R
    K = jnp.linalg.solve(S, H @ P).T          # P H^T S^-1
    delta = ErrorState.from_vec(K @ y)
    s_new = boxplus(s, delta)
    I_KH = jnp.eye(15) - K @ H
    P_joseph = I_KH @ P @ I_KH.T + K @ R @ K.T
    # Error angles now refer to s_new.q. Transform the full covariance,
    # including orientation cross-covariances, into that new error frame.
    G = construct_reset_jacobian(delta.theta)
    P_new = G @ P_joseph @ G.T
    P_new = 0.5 * (P_new + P_new.T)
    return s_new, P_new
