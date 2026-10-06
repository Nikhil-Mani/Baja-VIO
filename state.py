from typing import NamedTuple
import jax.numpy as jnp
from constants import *
from util import *

class State(NamedTuple):
    x: jnp.ndarray = jnp.zeros(3)
    dx: jnp.ndarray = jnp.zeros(3)
    q: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0])
    b_accel: jnp.ndarray = jnp.zeros(3)
    b_gyro: jnp.ndarray = jnp.zeros(3)

    def to_vec(self):
        return jnp.concatenate(self)  # (16,)

    @staticmethod
    def from_vec(v):
        return State(v[0:3], v[3:6], v[6:10], v[10:13], v[13:16])

class ErrorState(NamedTuple):
    x: jnp.ndarray = jnp.zeros(3)
    dx: jnp.ndarray = jnp.zeros(3)
    theta: jnp.ndarray = jnp.zeros(3)
    b_accel: jnp.ndarray = jnp.zeros(3)
    b_gyro: jnp.ndarray = jnp.zeros(3)

    def to_vec(self):
        return jnp.concatenate(self)  # (15,)

    @staticmethod
    def from_vec(v):
        return ErrorState(v[0:3], v[3:6], v[6:9], v[9:12], v[12:15])

class Input(NamedTuple):
    a: jnp.ndarray = jnp.zeros(3)
    w: jnp.ndarray = jnp.zeros(3)

# Error layout: [x, dx, theta, b_accel, b_gyro] -> P is 15x15 in this order
POS, VEL, THETA, BA, BG = (slice(0, 3), slice(3, 6), slice(6, 9),
                           slice(9, 12), slice(12, 15))

def boxplus(s: State, d: ErrorState) -> State:
    return State(
        x=s.x + d.x,
        dx=s.dx + d.dx,
        q=quat_mul(s.q, quat_from_rotvec(d.theta)),
        b_accel=s.b_accel + d.b_accel,
        b_gyro=s.b_gyro + d.b_gyro,
    )

def boxminus(s1: State, s2: State) -> ErrorState:
    dq = quat_mul(quat_conj(s2.q), s1.q)
    sign = jnp.where(dq[0] < 0, -1.0, 1.0)
    return ErrorState(
        x=s1.x - s2.x,
        dx=s1.dx - s2.dx,
        theta=2.0 * sign * dq[1:],
        b_accel=s1.b_accel - s2.b_accel,
        b_gyro=s1.b_gyro - s2.b_gyro,
    )
