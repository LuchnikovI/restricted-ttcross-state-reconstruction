import jax.numpy as jnp
from jax import random
from typing import List, Tuple
from copy import deepcopy
from mps_utils import random_normal_complex
from constants import sx, sy, sz, xyz0, xyz0_inv

from jax.config import config
config.update("jax_enable_x64", True)


def _im_ker_trace(
    ker: jnp.ndarray,
) -> jnp.ndarray:
    lb, _, rb = ker.shape
    ker = ker.reshape((lb, 2, 2, 2, 2, rb))
    ker = jnp.trace(ker, axis1=1, axis2=2)
    ker = jnp.trace(ker, axis1=1, axis2=2)
    return ker


def _gen_random_im_ker(
    subkey: jnp.ndarray,
    sqrt_rank_in: int,
    sqrt_rank_out: int,
    local_choi_rank: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    iso  = random_normal_complex(subkey, (2 * sqrt_rank_out * local_choi_rank, 2 * sqrt_rank_in))
    iso, _ = jnp.linalg.qr(iso)
    iso = iso.reshape((local_choi_rank, sqrt_rank_out, 2, 2, sqrt_rank_in))
    ker = jnp.einsum("qrijp,qsklt->rsikjlpt", iso, iso.conj())
    ker = ker.reshape((sqrt_rank_out * sqrt_rank_out, 16, sqrt_rank_in * sqrt_rank_in))
    return ker


def get_random_im(
    subkey: jnp.ndarray,
    sqrt_rank: int,
    local_choi_rank: int,
    time_steps: int,
) -> List[jnp.ndarray]:
    subkeys = random.split(subkey, time_steps)
    ker = _gen_random_im_ker(subkeys[0], 1, sqrt_rank, local_choi_rank)
    im = [ker]
    for subkey in subkeys[1:]:
        ker = _gen_random_im_ker(subkey, sqrt_rank, sqrt_rank, local_choi_rank)
        im = [ker] + im
    im[0] = jnp.trace(im[0].reshape((sqrt_rank, sqrt_rank, 16, -1)), axis1=0, axis2=1)[jnp.newaxis]
    return im


def get_id_im(
    time_steps: int,
) -> jnp.ndarray:
    im = []
    for _ in range(time_steps):
        ker = jnp.eye(4)
        ker = ker.reshape((1, 16, 1))
        im.append(ker)
    return im


def dens2bloch(dens: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([
        jnp.tensordot(sx, dens, axes=[[0, 1], [1, 0]]).real[jnp.newaxis],
        jnp.tensordot(sy, dens, axes=[[0, 1], [1, 0]]).real[jnp.newaxis],
        jnp.tensordot(sz, dens, axes=[[0, 1], [1, 0]]).real[jnp.newaxis]
    ], axis=0)


def get_dynamics(
    phis: List[jnp.ndarray],
    im_mps: List[jnp.ndarray],
    init_dens: jnp.ndarray,
) -> List[jnp.ndarray]:
    if len(phis) != len(im_mps):
        raise RuntimeError("Incorrect number of local quantum channels.")
    phis = deepcopy(phis)
    left_plugs = [jnp.ones((1,))]
    for i in range(len(im_mps) - 1):
        plug = left_plugs[-1]
        plug = jnp.tensordot(plug, _im_ker_trace(im_mps[i]), axes=1)
        plug /= jnp.linalg.norm(plug)
        left_plugs.append(plug)
    state = init_dens.reshape((1, -1))
    dens_list = []
    for i in range(len(im_mps) - 1, -1, -1):
        ker = im_mps[i]
        lb, _, rb = ker.shape
        ker = ker.reshape((lb, 4, 4, rb))
        state = jnp.einsum("qjip,pi->qj", ker, state)
        state = jnp.einsum("ij,qj->qi", phis.pop(), state)
        dens = jnp.einsum("q,qi->i", left_plugs.pop(), state)
        dens = dens.reshape((2, 2))
        norm = jnp.trace(dens)
        dens /= norm
        dens_list.append(dens)
        state /= norm
    return dens_list


def get_random_unitary_phi(
    key: jnp.ndarray,
    len: int,
) -> List[jnp.ndarray]:
    keys = random.split(key, len)
    phis = []
    for key in keys:
        phi = random.normal(key, (2, 2, 2))
        phi = phi[..., 0] + 1j * phi[..., 1]
        phi, _ = jnp.linalg.qr(phi)
        phi = phi[:, jnp.newaxis, :, jnp.newaxis] * phi.conj()[jnp.newaxis, :, jnp.newaxis, :]
        phis.append(phi.reshape((4, 4)))
    return phis


def get_chanel_explicit(im_mps: List[jnp.ndarray]) -> jnp.ndarray:
    dens = jnp.ones((1, 1, 1, 1, 1))
    for i, ker in enumerate(im_mps):
        left_bond = ker.shape[0]
        ker = ker.reshape((left_bond, 2, 2, 2, 2, -1))
        dens = jnp.einsum("ijklq,qmnrsp->imjnkrlsp", dens, ker)
        dens = dens.reshape((
            2 ** (i + 1),
            2 ** (i + 1),
            2 ** (i + 1),
            2 ** (i + 1),
            -1
        ))
    return dens[..., 0]


def im2corr(im_mps: List[jnp.ndarray]) -> List[jnp.ndarray]:
    corr_mps = []
    for ker in im_mps:
        lb, _, rb = ker.shape
        renorm_ker = ker / 2.
        renorm_ker = renorm_ker.reshape((lb, 2, 2, 2, 2, rb))
        corr_ker = jnp.einsum("qijklp,sji,rlk->qsrp", renorm_ker, xyz0, xyz0)
        corr_ker = corr_ker.reshape((lb, 16, rb))
        corr_mps.append(corr_ker)
    return corr_mps


def corr2im(corr_mps: List[jnp.ndarray]) -> List[jnp.ndarray]:
    im_mps = []
    for ker in corr_mps:
        lb, _, rb = ker.shape
        renorm_ker = ker * 2.
        renorm_ker = renorm_ker.reshape((lb, 4, 4, rb))
        im_ker = jnp.einsum("qsrp,jis,lkr->qijklp", renorm_ker, xyz0_inv, xyz0_inv)
        im_ker = im_ker.reshape((lb, 16, rb))
        im_mps.append(im_ker)
    return im_mps
