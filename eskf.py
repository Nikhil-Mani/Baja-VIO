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
    R = quat_to_R(s.q)
    w = u_w - s.b_gyro
    v_b = R.T @ s.dx
    d_out, d_in = ackermann_angles(params, delta)
    angles = jnp.array([d_in, d_out, 0.0, 0.0])
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles), jnp.zeros(4)], axis=1)
    v_w = v_b + jnp.cross(w, wheel_pos)
    v_long = jnp.sum(v_w * dirs, axis=1)
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

def construct_Q(dt, sig_a, sig_g, sig_ba, sig_bg):
    # simple diagonal process noise for now
    d = jnp.concatenate([jnp.zeros(3),
                         jnp.full(3, sig_a**2 * dt),
                         jnp.full(3, sig_g**2 * dt),
                         jnp.full(3, sig_ba**2 * dt),
                         jnp.full(3, sig_bg**2 * dt)])
    return jnp.diag(d)

@jax.jit
def predict(s, P, u, Q, dt):
    F = construct_F(s, u, dt)
    return f(s, u, dt), F @ P @ F.T + Q

@jax.jit
def update(s, P, z, R, *h_args):
    H = construct_H(s, *h_args)
    y = z - h(s, *h_args)
    S = H @ P @ H.T + R
    K = jnp.linalg.solve(S, H @ P).T          # P H^T S^-1
    s_new = boxplus(s, ErrorState.from_vec(K @ y))
    I_KH = jnp.eye(15) - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
    return s_new, P_new
