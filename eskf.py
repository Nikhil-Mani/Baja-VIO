import jax
import jax.numpy as jnp
import numpy as np
from constants import *
from util import *
from state import *

# nonlinear nominal dynamics
def f(s: State, u_left: Input, u_right: Input, dt) -> State:
    # Use the average from two IMU samples using midpoint
    accel = .5 * (u_left.a + u_right.a) - s.b_accel
    omega = .5 * (u_left.w + u_right.w) - s.b_gyro
    q_half = quat_mul(s.q, quat_from_rotvec(omega * dt / 2))
    # Transform acc using orientation halfway through the interval.
    a = quat_to_R(q_half) @ accel + g
    q = quat_mul(s.q, quat_from_rotvec(omega * dt))

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
def construct_F(s, u_left, u_right, dt):
    nominal = f(s, u_left, u_right, dt)
    def err(v):
        # boxplus is just generalized plus to combine the state and error state
        vec = boxplus(s, ErrorState.from_vec(v))
        # get difference between the nominal and perturbed state to differentiate (del_x)
        return boxminus(f(vec, u_left, u_right, dt), nominal).to_vec()

    return jax.jacfwd(err)(jnp.zeros(15))

def construct_H(s, *h_args):
    def err(v):
        return h(boxplus(s, ErrorState.from_vec(v)), *h_args)

    return jax.jacfwd(err)(jnp.zeros(15))

def construct_Q(s, u_left, u_right, dt, sig_a, sig_g, sig_ba, sig_bg):

    nominal = f(s, u_left, u_right, dt)
    def err(n):  # left accel/gyro (6), right accel/gyro (6)
        # add noise to values 1 and 2
        left = Input(a=u_left.a + n[:3], w=u_left.w + n[3:6])
        right = Input(a=u_right.a + n[6:9], w=u_right.w + n[9:12])
        return boxminus(f(s, left, right, dt), nominal).to_vec()
    # differentaite noisy F wrt the noise
    # represents difference in state given a noise pertubation!
    Fi = jax.jacfwd(err)(jnp.zeros(12))
    sample_variance = jnp.tile(jnp.repeat(jnp.array([sig_a**2 / dt,
                                                    sig_g**2 / dt]), 3), 2)
    Q = (Fi * sample_variance[None, :]) @ Fi.T
    Q = Q.at[BA, BA].add(jnp.eye(3) * sig_ba**2 * dt)   # bias random walks
    Q = Q.at[BG, BG].add(jnp.eye(3) * sig_bg**2 * dt)
    return Q

def construct_reset_jacobian(delta_theta):
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
    # essentially just the formula from Sola
    rotation_reset = jnp.eye(3) - a * angle_skew + b * (angle_skew @ angle_skew)
    return jnp.eye(15).at[THETA, THETA].set(rotation_reset)

@jax.jit
def predict(s, P, u_left, u_right, dt, *noise):
    # Predict state/covariance over a positive-dt interval with two IMU endpoints.

    F = construct_F(s, u_left, u_right, dt)
    Q = construct_Q(s, u_left, u_right, dt, *noise)
    return f(s, u_left, u_right, dt), F @ P @ F.T + Q

@jax.jit
def update(s, P, z, R, *h_args):
    H = construct_H(s, *h_args)
    y = z - h(s, *h_args)
    S = H @ P @ H.T + R
    K = jnp.linalg.solve(S, H @ P).T          # P H^T S^-1
    # Block only wheel corrections; retain state uncertainty and IMU propagation.
    K = K.at[jnp.array([2, 5, 11]), :].set(0.)
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


def initialize(s: State, *, accel_bias=None, gyro_bias=None,
               accel_bias_std=.1, gyro_bias_std=.01,
               position_std=.01, velocity_std=.1, attitude_std=.01):
    def vector(value, name, nonnegative=False):
        value = np.asarray(value, dtype=float)
        return jnp.asarray(value)

    calibrated = s._replace(
        b_accel=s.b_accel if accel_bias is None else vector(accel_bias, 'accel_bias'),
        b_gyro=s.b_gyro if gyro_bias is None else vector(gyro_bias, 'gyro_bias'))
    std = jnp.concatenate([vector(v, name, True) for v, name in
                           ((position_std, 'position_std'),
                            (velocity_std, 'velocity_std'),
                            (attitude_std, 'attitude_std'),
                            (accel_bias_std, 'accel_bias_std'),
                            (gyro_bias_std, 'gyro_bias_std'))])
    return calibrated, jnp.diag(std**2)
