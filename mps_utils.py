import jax.numpy as jnp
from jax import jit, random
from typing import List, Tuple

sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
s0 = jnp.eye(2, dtype=jnp.complex64)
xyz0 = jnp.concatenate([s0[jnp.newaxis], sx[jnp.newaxis], sy[jnp.newaxis], sz[jnp.newaxis]], axis=0)

def conj(tt: List[jnp.ndarray]) -> List[jnp.ndarray]:
    tt_conj = []
    for ker in tt:
        tt_conj.append(ker.conj())
    return tt_conj

def _push_r_backward(
    ker: jnp.ndarray,
    r: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    _, dim, right_bond = ker.shape
    left_bond = r.shape[0]
    ker = jnp.tensordot(r, ker, axes=1)
    ker = ker.reshape((-1, right_bond))
    ker, r = jnp.linalg.qr(ker)
    ker = ker.reshape((left_bond, dim, -1))
    return ker, r

def set_to_forward_canonical(
    mps: List[jnp.ndarray],
) -> jnp.ndarray:
    r = jnp.eye(mps[0].shape[0])
    lognorm = 0.
    for i, ker in enumerate(mps):
        ker, r = _push_r_backward(ker, r)
        norm = jnp.linalg.norm(r)
        r /= norm
        lognorm += jnp.log(norm)
        mps[i] = ker
    mps[-1] = jnp.tensordot(mps[-1], r, axes=1)
    return lognorm

def random_corr(
    subkey: jnp.ndarray,
    qubits_num: int,
    rank: int,
) -> List[jnp.ndarray]:
    subkeys = random.split(subkey, qubits_num)
    left_bonds = [1] + (qubits_num - 1) * [rank]
    right_bonds = (qubits_num - 1) * [rank] + [1]
    mps = []
    for (subkey, left_bond, right_bond) in zip(subkeys, left_bonds, right_bonds):
        ker = random.normal(subkey, (left_bond, 2, right_bond, 2))
        ker = ker[..., 0] + 1j * ker[..., 1]
        mps.append(ker)
    set_to_forward_canonical(mps)
    obs = []
    for ker in mps:
        left_bond = ker.shape[0]
        ker = jnp.einsum("qip,mjn,kij->qmkpn", ker, ker.conj(), xyz0)
        ker = ker.reshape((left_bond ** 2, 4, -1))
        obs.append(ker)
    return obs

def dot(
    lhs: List[jnp.ndarray],
    rhs: List[jnp.ndarray],
) -> jnp.array:
    log_norm = 0
    l = jnp.ones((1, 1))
    for (lhs_ker, rhs_ker) in zip(lhs, rhs):
        l = jnp.tensordot(l, lhs_ker, axes=1)
        l = jnp.tensordot(rhs_ker, l, axes=[[0, 1], [0, 1]])
        norm = jnp.linalg.norm(l)
        l /= norm
        log_norm += jnp.log(norm)
    return jnp.exp(log_norm) * l[0, 0]

def check_indices(
    indices: jnp.ndarray,
) -> int:
    return (indices > 0).sum(1).max()

@jit
def log_eval(
    tt: List[jnp.ndarray],
    indices: jnp.ndarray,
) -> Tuple[jnp.array, jnp.array]:
    log_norms = jnp.zeros((indices.shape[0],))
    l = jnp.ones((indices.shape[0], 1))
    for (slice, ker) in zip(indices.T, tt):
        l = jnp.einsum("bi,bij->bj", l, ker[:, slice, :].transpose((1, 0, 2)))
        norm = jnp.linalg.norm(l, axis=1, keepdims=True)
        l /= norm
        log_norms += jnp.log(norm[:, 0])
    return log_norms, l[:, 0]