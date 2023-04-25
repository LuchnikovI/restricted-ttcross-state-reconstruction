import jax.numpy as jnp
from exact_im_utils import (
    delta_one_mod_a_minus_b,
    get_mps_im,
)
from im_utils import get_chanel_explicit

from jax.config import config
config.update("jax_enable_x64", True)

def test_delta_one_mod_a_minus_b():
    matrix = delta_one_mod_a_minus_b(5)
    true_matrix = jnp.array([
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ])
    assert((matrix == true_matrix).all())

def test_get_mps_im():
    for m in range(10, 20):
        mps = get_mps_im(m, 15, 5)
        assert(len(mps) == 10)
        phi = get_chanel_explicit(mps)
        assert((jnp.abs(jnp.trace(phi, axis1=0, axis2=1) - jnp.eye(32)) < 1e-5).all())
        phi = phi.transpose((0, 2, 1, 3))
        choi = phi.reshape((32 * 32, 32 * 32))
        assert((jnp.linalg.eigvalsh(choi) > -1e-5).all())
    for n in range(10, 20):
        mps = get_mps_im(15, n, 5)
        assert(len(mps) == 10)
        phi = get_chanel_explicit(mps)
        assert((jnp.abs(jnp.trace(phi, axis1=0, axis2=1) - jnp.eye(32)) < 1e-5).all())
        phi = phi.transpose((0, 2, 1, 3))
        choi = phi.reshape((32 * 32, 32 * 32))
        assert((jnp.linalg.eigvalsh(choi) > -1e-5).all())
