import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
s0 = jnp.eye(2, dtype=jnp.complex128)
xyz = jnp.concatenate([sx[jnp.newaxis], sy[jnp.newaxis], sz[jnp.newaxis]], axis=0)
xyz0 = jnp.concatenate([s0[jnp.newaxis], sx[jnp.newaxis], sy[jnp.newaxis], sz[jnp.newaxis]], axis=0)
xyz0_inv = jnp.linalg.inv(xyz0.reshape((4, 4))).reshape((2, 2, 4))