import jax
jax.config.update("jax_enable_x64", True)  # your Ackermann fit needs this until you refit
import jax.numpy as jnp
from typing import NamedTuple

ackermann_mapping_params: dict = {
        "a": -0.0006468948801,
        "b": -4.538654441,
        "c": -21232.95668,
        "d": -49662479.71,
        "h": 49662491.96,
        "l": 0.0004275451321,
        }
g = jnp.array([0, 0, -9.81])

class CarParams(NamedTuple):
    T: float = 10.0 # Placeholder
    L: float = 10.0 # Placeholder
    r_w: float = 0.3 # Placeholder
    ack_per: dict = ackermann_mapping_params
