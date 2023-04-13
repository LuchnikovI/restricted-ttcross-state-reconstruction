import jax.numpy as jnp
from jax import jit, random
from typing import List, Tuple


sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
s0 = jnp.eye(2, dtype=jnp.complex64)
xyz0 = jnp.concatenate([s0[jnp.newaxis], sx[jnp.newaxis], sy[jnp.newaxis], sz[jnp.newaxis]], axis=0)
xyz0_inv = jnp.linalg.inv(xyz0.reshape((4, 4))).reshape((2, 2, 4))


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


def _push_orth_center_forward(
    ker: jnp.ndarray,
    u: jnp.ndarray,
    spec: jnp.ndarray,
    rank: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    ker = jnp.tensordot(ker, u, axes=1)
    q = ker * spec
    right_bond = q.shape[0]
    q = q.reshape((right_bond, -1))
    u, s, _ = jnp.linalg.svd(q, full_matrices=False)
    u, s = u[:, :rank], s[:rank]
    ker = jnp.tensordot(u.T.conj(), ker, axes=1)
    return ker, u, s


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


def truncate_forward_canonical(
    inp_mps: List[jnp.ndarray],
    rank: int,
):
    u = jnp.eye(inp_mps[-1].shape[-1])
    spec = jnp.ones((inp_mps[-1].shape[-1],))
    for i, ker in enumerate(reversed(inp_mps)):
        ker, u, spec = _push_orth_center_forward(ker, u, spec, rank)
        inp_mps[len(inp_mps) - i - 1] = ker


def random_pure_state(
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
    return mps


def pure_state2dens(
    mps: List[jnp.ndarray],
) -> List[jnp.ndarray]:
    dens_mps = []
    for ker in mps:
        left_bond = ker.shape[0]
        dens_ker = jnp.einsum("qip,mjn->qmijpn", ker, ker.conj())
        dens_ker = dens_ker.reshape((left_bond ** 2, 4, -1))
        dens_mps.append(dens_ker)
    return dens_mps


def dens2corr(
    dens_mps: List[jnp.ndarray]
) -> List[jnp.ndarray]:
    corr_mps = []
    for dens_ker in dens_mps:
        left_bond = dens_ker.shape[0]
        dens_ker = dens_ker.reshape((left_bond, 2, 2, -1))
        ker = jnp.einsum("qijp,kji->qkp", dens_ker, xyz0)
        corr_mps.append(ker)
    return corr_mps


def corr2dens(
    corr_mps: List[jnp.ndarray]
) -> List[jnp.ndarray]:
    dens_mps = []
    for corr_ker in corr_mps:
        dens_ker = jnp.einsum("qkp,jik->qijp", corr_ker, xyz0_inv)
        left_bond = dens_ker.shape[0]
        dens_ker = dens_ker.reshape((left_bond, 4, -1))
        dens_mps.append(dens_ker)
    return dens_mps


def fidelity(
    dens_mps: List[jnp.ndarray],
    mps: List[jnp.ndarray],
) -> float:
    log_fid = 0
    lhs = jnp.ones((1, 1, 1))
    for mps_ker, dens_ker in zip(mps, dens_mps):
        left_bond = dens_ker.shape[0]
        dens_ker = dens_ker.reshape((left_bond, 2, 2, -1))
        lhs = jnp.einsum("ijk,ipa->apjk", lhs, mps_ker)
        lhs = jnp.einsum("apjk,jqpb->abqk", lhs, dens_ker)
        lhs = jnp.einsum("abqk,kqc->abc", lhs, mps_ker.conj())
        norm = jnp.linalg.norm(lhs)
        lhs /= norm
        log_fid += jnp.log(norm)
    return jnp.exp(log_fid)


def dens_trace(dens_mps: List[jnp.ndarray]) -> float:
    log_norm = 0
    lhs = jnp.ones((1,))
    for dens_ker in dens_mps:
        left_bond = dens_ker.shape[0]
        dens_ker = dens_ker.reshape((left_bond, 2, 2, -1))
        lhs = jnp.tensordot(lhs, jnp.einsum("qiip->qp", dens_ker), axes=1)
        norm = jnp.linalg.norm(lhs)
        lhs /= norm
        log_norm += jnp.log(norm)
    return (jnp.exp(log_norm) * lhs[0])


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
