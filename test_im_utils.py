import jax.numpy as jnp
from jax import random
from im_utils import (
    _gen_random_im_ker,
    get_random_unitary_phi,
    get_dynamics,
    get_random_im,
    get_chanel_explicit,
    im2corr,
    corr2im,
)
from mps_utils import eval, log_eval
import pytest
from jax.config import config
from mps_utils import set_to_forward_canonical, dot, conj

config.update("jax_enable_x64", True)

key = random.PRNGKey(42)


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("local_choi_rank", [1, 2, 4])
@pytest.mark.parametrize("sqrt_rank_in,sqrt_rank_out",
[
    (1, 1),
    (2, 2),
    (4, 4),
    (1, 4),
])
def test_gen_random_im_kers(
    subkey: jnp.ndarray,
    local_choi_rank: int,
    sqrt_rank_in: int,
    sqrt_rank_out: int,
):
    phi = _gen_random_im_ker(subkey, sqrt_rank_in, sqrt_rank_out, local_choi_rank)
    phi = phi.reshape((sqrt_rank_out, sqrt_rank_out, 2, 2, 2, 2, sqrt_rank_in, sqrt_rank_in))
    tr_phi = jnp.einsum("qqppijkl->ikjl", phi)
    tr_phi = tr_phi.reshape((2 * sqrt_rank_in, 2 * sqrt_rank_in))
    assert((jnp.abs(tr_phi - jnp.eye(2 * sqrt_rank_in)) < 1e-5).all())
    phi = phi.transpose((0, 2, 4, 6, 1, 3, 5, 7)).reshape((4 * sqrt_rank_in * sqrt_rank_out, 4 * sqrt_rank_in * sqrt_rank_out))
    assert((jnp.abs(phi - phi.T.conj()) < 1e-5).all())
    lmbd = jnp.linalg.eigvalsh(phi)
    assert((lmbd > -1e-5).all())


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("time_steps",[1, 3, 5])
@pytest.mark.parametrize("local_choi_rank",[1, 2, 4])
@pytest.mark.parametrize("sqrt_rank",[1, 5, 10])
def test_get_random_im_and_get_channel_explicit(
    subkey: jnp.ndarray,
    time_steps: int,
    local_choi_rank: int,
    sqrt_rank: int,
):
    dim = 2 ** time_steps
    im = get_random_im(subkey, sqrt_rank, local_choi_rank, time_steps)
    assert(len(im) == time_steps)
    phi = get_chanel_explicit(im)
    assert((jnp.abs(jnp.trace(phi, axis1=0, axis2=1) - jnp.eye(dim)) < 1e-5).all())
    phi = phi.transpose((0, 2, 1, 3))
    choi = phi.reshape((dim ** 2, dim ** 2))
    assert((jnp.linalg.eigvalsh(choi) > -1e-5).all())

@pytest.mark.parametrize("subkey", random.split(key, 2))
def test_get_random_unitary_phi(
    subkey: jnp.ndarray
):
    phis = get_random_unitary_phi(subkey, 10)
    for phi in phis:
        phi = phi.reshape((2, 2, 2, 2))
        id = jnp.einsum("iiqp->qp", phi)
        assert((jnp.abs(id - jnp.eye(2)) < 1e-5).all())


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("time_steps",[1, 5, 20])
@pytest.mark.parametrize("sqrt_rank",[1, 5, 10])
@pytest.mark.parametrize("local_choi_rank",[1, 2, 4])
def test_get_dynamics(
    subkey: jnp.ndarray,
    time_steps: int,
    sqrt_rank: int,
    local_choi_rank: int,
):
    key, subkey = random.split(subkey)
    mps = get_random_im(subkey, sqrt_rank, local_choi_rank, time_steps)
    key, subkey = random.split(key)
    phis = get_random_unitary_phi(subkey, time_steps)
    for dens in get_dynamics(phis, mps, jnp.array([1., 0., 0., 0.])):
        print(dens)
        assert((jnp.abs(jnp.trace(dens) - 1.) < 1e-5).all())
        assert((jnp.abs(dens - dens.conj().T) < 1e-5).all())
        assert((jnp.linalg.eigvalsh(dens) > -1e-5).all())


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("sqrt_rank",[1, 5, 10])
@pytest.mark.parametrize("local_choi_rank",[1, 2, 4])
@pytest.mark.parametrize("time_steps",[1, 5, 20])
def test_im2corr_corr2im(
    subkey: jnp.ndarray,
    sqrt_rank: int,
    local_choi_rank: int,
    time_steps: int,
):
    im = get_random_im(subkey, sqrt_rank, local_choi_rank, time_steps)
    corr = im2corr(im)
    _, subkey = random.split(subkey)
    assert(jnp.abs(eval(corr, jnp.zeros((1, time_steps), dtype=jnp.uint)) - 1.) < 1e-5)
    #log_abs, _ = log_eval(corr, random.categorical(subkey, jnp.ones((16,)), shape = (10, time_steps)))
    #assert((log_abs < 0.).all())
    im_from_corr = corr2im(corr)
    log_norm_im = set_to_forward_canonical(im)
    log_norm_im_from_corr = set_to_forward_canonical(im_from_corr)
    assert(jnp.abs(log_norm_im - log_norm_im_from_corr) < 1e-5)
    assert((dot(im, conj(im_from_corr)) - 1.) < 1e-5)
