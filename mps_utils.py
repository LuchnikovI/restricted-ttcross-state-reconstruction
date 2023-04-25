import jax.numpy as jnp
from jax import jit, random
from typing import List, Tuple, Iterable
from constants import xyz0, xyz0_inv


def random_normal_complex(
    subkey: jnp.ndarray,
    shape: Iterable[int],
) -> jnp.ndarray:
    val = random.normal(subkey, shape + (2,))
    val = val[..., 0] + 1j * val[..., 1]
    return val


def random_normal_mps(
    subkey: jnp.ndarray,
    size: int,
    rank: int,
) -> List[jnp.ndarray]:
    if size == 0:
        return []
    elif size == 1:
        mps = [random_normal_complex(subkey, (1, 2, 1))]
        return mps
    else:
        shapes = [(1, 2, rank)] + (size - 2) * [(rank, 2, rank)] + [(rank, 2, 1)]
        subkeys = random.split(subkey, size)
        mps = [random_normal_complex(subkey, shape) for (subkey, shape) in zip(subkeys, shapes)]
        return mps


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
    r = jnp.eye(1)
    lognorm = 0.
    if len(mps) == 0:
        return lognorm
    for i, ker in enumerate(mps):
        ker, r = _push_r_backward(ker, r)
        norm = jnp.linalg.norm(r)
        r /= norm
        lognorm += jnp.log(norm)
        mps[i] = ker
    mps[-1] = jnp.tensordot(mps[-1], r, axes=1)
    return lognorm


def log_dot(
    lhs: List[jnp.ndarray],
    rhs: List[jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    log_norm = 0
    l = jnp.ones((1, 1))
    for (lhs_ker, rhs_ker) in zip(lhs, rhs):
        l = jnp.tensordot(l, lhs_ker, axes=1)
        l = jnp.tensordot(rhs_ker, l, axes=[[0, 1], [0, 1]])
        norm = jnp.linalg.norm(l)
        l /= norm
        log_norm += jnp.log(norm)
    return log_norm, l[0, 0]


def dot(
    lhs: List[jnp.ndarray],
    rhs: List[jnp.ndarray],
) -> jnp.ndarray:
    log_norm, phase = log_dot(lhs, rhs)
    return jnp.exp(log_norm) * phase


def log_sum(
    mps: List[jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    log_norm = 0.
    plug = jnp.ones((1, 1))
    for ker in mps:
        ker = ker.sum(1)
        plug = jnp.tensordot(plug, ker, axes=1)
        norm = jnp.linalg.norm(plug)
        plug /= norm
        log_norm += jnp.log(norm)
    return log_norm, plug[0, 0]


def sum(
    mps: List[jnp.ndarray],
) -> jnp.ndarray:
    log_norm, phase = log_sum(mps)
    return jnp.exp(log_norm) * phase


def conj(tt: List[jnp.ndarray]) -> List[jnp.ndarray]:
    tt_conj = []
    for ker in tt:
        tt_conj.append(ker.conj())
    return tt_conj


def _push_orth_center_forward(
    ker: jnp.ndarray,
    u: jnp.ndarray,
    spec: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    left_bond, dim, _ = ker.shape
    ker = jnp.tensordot(ker, u, axes=1)
    ker = ker * spec
    ker = ker.reshape((left_bond, -1))
    u, s, ker = jnp.linalg.svd(ker, full_matrices=False)
    ker = ker.reshape((ker.shape[0], dim, -1))
    return ker, u, s


def _set_rank(spec: jnp.ndarray, eps: float) -> jnp.ndarray:
    cum_sq_sum = jnp.cumsum(spec[::-1] ** 2)
    sq_sum = (spec ** 2).sum()
    trsh = (jnp.sqrt(cum_sq_sum / sq_sum) > eps).sum()
    return trsh


def truncate_forward_canonical(
    mps: List[jnp.ndarray],
    rank: int,
):
    if len(mps) == 0:
        return
    u = jnp.eye(mps[-1].shape[-1])
    spec = jnp.ones((mps[-1].shape[-1],))
    for i, ker in enumerate(reversed(mps)):
        ker, u, spec = _push_orth_center_forward(ker, u, spec)
        trsh = min(rank, spec.shape[0])
        ker, u, spec = ker[:trsh], u[:, :trsh], spec[:trsh]
        mps[len(mps) - i - 1] = ker
    mps[0] = jnp.tensordot(u, mps[0], axes=1)


def truncate_forward_canonical_by_error(
    mps: List[jnp.ndarray],
    error: float,
):
    if len(mps) == 0:
        return
    u = jnp.eye(mps[-1].shape[-1])
    spec = jnp.ones((mps[-1].shape[-1],))
    for i, ker in enumerate(reversed(mps)):
        ker, u, spec = _push_orth_center_forward(ker, u, spec)
        rank = _set_rank(spec, error)
        trsh = min(rank, spec.shape[0])
        ker, u, spec = ker[:trsh], u[:, :trsh], spec[:trsh]
        mps[len(mps) - i - 1] = ker
    mps[0] = jnp.tensordot(u, mps[0], axes=1)

def mul_by_log_scalar(
    mps: List[jnp.ndarray],
    log_scalar: float,
):
    size = len(mps)
    if size == 0:
        return
    local_mul = jnp.exp(log_scalar / size)
    for i in range(size):
        mps[i] = mps[i] * local_mul


def truncate_mps_by_error(
    mps: List[jnp.ndarray],
    error: float,
):
    log_norm = set_to_forward_canonical(mps)
    truncate_forward_canonical_by_error(mps, error)
    mul_by_log_scalar(mps, log_norm)


def truncate_mps(
    mps: List[jnp.ndarray],
    rank: int,
):
    log_norm = set_to_forward_canonical(mps)
    truncate_forward_canonical(mps, rank)
    mul_by_log_scalar(mps, log_norm)


def _ker_trace(ker: jnp.ndarray) -> jnp.ndarray:
    left_bond = ker.shape[0]
    ker = ker.reshape((left_bond, 2, 2, -1))
    ker = jnp.trace(ker, axis1=1, axis2=2)
    return ker


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
        ker = random_normal_complex(subkey, (left_bond, 2, right_bond))
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
        lhs = jnp.tensordot(lhs, _ker_trace(dens_ker), axes=1)
        norm = jnp.linalg.norm(lhs)
        lhs /= norm
        log_norm += jnp.log(norm)
    return (jnp.exp(log_norm) * lhs[0])


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
        l = jnp.einsum("bi,ibj->bj", l, ker[:, slice, :])
        norm = jnp.linalg.norm(l, axis=1, keepdims=True)
        l /= norm
        log_norms += jnp.log(norm[:, 0])
    return log_norms, l[:, 0]


@jit
def eval(
    tt: List[jnp.ndarray],
    indices: jnp.ndarray,
) -> jnp.ndarray:
    log_norm, phase = log_eval(tt, indices)
    return jnp.exp(log_norm) * phase
