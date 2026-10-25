import jax.numpy as jnp
from constants import *

def skew(v):
    return jnp.array(
            [[0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]])

# Q format is [w, x, y, z]
def quat_mul(q1, q2):
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    return jnp.concatenate(
            [jnp.array([w1 * w2 - v1 @ v2]),
            w1 * v2 + w2 * v1 + jnp.cross(v1, v2)])
# Inverse the vector component, keep scalar the same
def quat_conj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def quat_from_rotvec(theta):
    th2 = theta @ theta
    small = th2 < 1e-8
    # if small, approximate as 1 for safe division
    safe_th2 = jnp.where(small, 1.0, th2)
    th = jnp.sqrt(safe_th2)
    # use taylor series approximations if small
    cos = jnp.where(small, 1.0 - th2 / 8.0, jnp.cos(th / 2))
    sin = jnp.where(small, 0.5 - th2 / 48.0, jnp.sin(th / 2) / th)

    return jnp.concatenate([cos[None], sin * theta])

def quat_to_R(q):
    w, v = q[0], q[1:]
    # Rodrigues' formula (but for quats)
    return (w * w - v @ v) * jnp.eye(3) + 2 * jnp.outer(v, v) + 2 * w * skew(v)

def ackermann_percent(p, delta):
    """Evaluate the fitted Ackermann percentage at a steering angle in degrees."""
    # Desmos fitting yielded this curve with R^2 of .99
    return (p["a"] * delta**3 + p["b"] * delta**2 + p["c"] * delta
            + p["d"] + p["h"] * jnp.exp(p["l"] * delta))

def ackermann_angles(params, delta):
    """Return outer and inner wheel angles in radians for delta in radians."""
    # standard formula is arctan(L / (L/tan(d) +- T/2))
    # doesnt work for delta = 0, so rewrite with tan in numerator
    tan = jnp.tan(delta)
    # The fit uses degrees and returns percent; geometry uses radians.
    s = ackermann_percent(params.ack_per, jnp.rad2deg(delta)) / 100.0
    delta_out = s * jnp.arctan(params.L * tan / (params.L + 0.5 * params.T * tan))
    delta_in = s * jnp.arctan(params.L * tan / (params.L - 0.5 * params.T * tan))
    return delta_out, delta_in
