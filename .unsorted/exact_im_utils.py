import jax.numpy as jnp
from jax import random
from typing import List
from mps_utils import _ker_trace
from copy import deepcopy

from jax.config import config
config.update("jax_enable_x64", True)


def delta_one_mod_a_minus_b(m: int) -> jnp.ndarray:
    bond_range = jnp.arange(m)
    diff = bond_range.reshape((m, 1)) - bond_range.reshape((1, m))
    return jnp.mod(diff, m) == 1


def get_a(n: int, m: int) -> jnp.ndarray:
    delta_ab = delta_one_mod_a_minus_b(m)
    delta_ab_tr = delta_ab.T
    k = (float(n) / float(m)) * jnp.pi
    diag = jnp.cos(2. * k * jnp.arange(m))
    diag_shifted = jnp.cos(2. * k * jnp.arange(1, m + 1))
    a_00 = jnp.diag(jnp.cos(2. * k * jnp.arange(m)))
    a_11 = a_00
    a_01 = delta_ab * diag.reshape((1, -1))
    a_10 = delta_ab_tr * diag_shifted.reshape((-1, 1))
    a = jnp.concatenate(
        [
            a_00[..., jnp.newaxis],
            a_01[..., jnp.newaxis],
            a_10[..., jnp.newaxis],
            a_11[..., jnp.newaxis],
        ],
        axis=-1,
    )
    a = a.reshape((m, m, 4))
    a = a.transpose((0, 2, 1))
    return a / 2.


def get_b(n: int, m: int) -> jnp.ndarray:
    k = (float(n) / float(m)) * jnp.pi
    diag = jnp.exp(2. * k * 1j * jnp.arange(m))
    diag_conj = diag.conj()
    b_00 = jnp.diag(diag)
    b_11 = jnp.diag(diag_conj)
    b_01 = jnp.zeros((m, m))
    b_10 = jnp.zeros((m, m))
    b = jnp.concatenate(
        [
            b_00[..., jnp.newaxis],
            b_01[..., jnp.newaxis],
            b_10[..., jnp.newaxis],
            b_11[..., jnp.newaxis],
        ],
        axis=-1,
    )
    b = b.reshape((m, m, 4))
    b = b.transpose((0, 2, 1))
    return b


def get_mps_im(m: int, n: int, t: int) -> List[jnp.ndarray]:
    a = get_a(n, m)
    b = get_b(n, m)
    omega = a.sum(axis=-1, keepdims=1)
    v = a[0][jnp.newaxis]
    kers = []
    kers.append(v)
    for _ in range(t-2):
        kers.append(b)
        kers.append(a)
    kers.append(b)
    kers.append(omega)
    kers.append(jnp.eye(2).reshape((1, 4, 1)))
    return kers
