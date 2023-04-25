import jax.numpy as jnp
from constants import xyz0, xyz0_inv

def test_xyz0_and_xyz0_inv():
    prod = jnp.tensordot(xyz0_inv, xyz0, axes=1)
    assert((jnp.abs(prod.reshape((4, 4)) - jnp.eye(4)) < 1e-5).all())