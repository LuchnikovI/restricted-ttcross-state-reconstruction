from copy import deepcopy
from jax import random
import jax.numpy as jnp
from mps_utils import (
    random_normal_complex,
    random_normal_mps,
    _push_r_backward,
    set_to_forward_canonical,
    _push_orth_center_forward,
    truncate_forward_canonical,
    truncate_forward_canonical_by_error,
    truncate_mps_by_error,
    truncate_mps,
    mul_by_log_scalar,
    _set_rank,
    random_pure_state,
    pure_state2dens,
    dens_trace,
    dens2corr,
    corr2dens,
    fidelity,
    conj,
    log_dot,
    dot,
    sum,
    eval,
)
from typing import Iterable, Tuple
import pytest

key = random.PRNGKey(42)


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("shape", [(1,), (2,), (3, 2, 1, 2)])
def test_random_normal_complex(
    subkey: jnp.ndarray,
    shape: Iterable[int],
):
    assert(random_normal_complex(subkey, shape).shape == shape)


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("size", [0, 1, 5, 40])
@pytest.mark.parametrize("rank", [1, 5, 25])
def test_random_normal_mps(
    subkey: jnp.ndarray,
    size: int,
    rank: int,
):
    mps = random_normal_mps(
        subkey,
        size,
        rank,
    )
    assert(len(mps) == size)
    if size == 1:
        assert(mps[0].shape == (1, 2, 1))
    elif size >= 2:
        assert(mps[0].shape == (1, 2, rank))
        assert(mps[-1].shape == (rank, 2, 1))
        for ker in mps[1:-1]:
            assert(ker.shape == (rank, 2, rank))


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("ker_shape,r_shape", [
    ((1, 1, 1), (3, 1)),
    ((9, 2, 3), (5, 9)),
    ((3, 2, 9), (3, 3)),
    ((4, 15, 3), (2, 4)),
])
def test_push_r_backward(
    subkey: jnp.ndarray,
    ker_shape: Tuple[int, int, int],
    r_shape: Tuple[int, int],
):
    subkey1, subkey2 = random.split(subkey, 2)
    ker = random_normal_complex(subkey1, ker_shape)
    r = random_normal_complex(subkey2, r_shape)
    new_ker, new_r = _push_r_backward(ker, r)
    assert(r.shape[0] == new_ker.shape[0])
    assert(new_r.shape[1] == ker.shape[2])
    assert(min(new_ker.shape[0] * new_ker.shape[1], new_r.shape[1]) == new_r.shape[0])
    assert(new_ker.shape[2] == new_r.shape[0])
    rker = jnp.tensordot(r, ker, axes=1)
    kerr = jnp.tensordot(new_ker, new_r, axes=1)
    assert((jnp.abs(rker - kerr) < 1e-5).all())
    kerker = jnp.tensordot(new_ker, jnp.conj(new_ker), axes=[[1, 0], [1, 0]])
    assert(jnp.abs(kerker - jnp.eye(new_ker.shape[2])) < 1e-5).all()


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("size", [0, 1, 2, 3, 25])
@pytest.mark.parametrize("rank", [1, 5])
def test_set_to_forward_canonical_and_log_dot(
    size: int,
    subkey: jnp.ndarray,
    rank: int,
):
    mps = random_normal_mps(subkey, size, rank)
    mps_copy = deepcopy(mps)
    log_norm = set_to_forward_canonical(mps)
    ldot, _ = log_dot(mps_copy, conj(mps_copy))
    assert(jnp.abs(ldot / 2 - log_norm) < 1e-5)
    ldot, _ = log_dot(mps, conj(mps))
    assert(jnp.abs(jnp.exp(ldot) - 1.) < 1e-5)
    ldot, _ = log_dot(mps_copy, conj(mps))
    assert(jnp.abs(ldot - log_norm) < 1e-5)
    for ker in mps:
        kerker = jnp.tensordot(ker, ker.conj(), axes=[[0, 1], [0, 1]])
        assert((jnp.abs(kerker - jnp.eye(kerker.shape[0])) < 1e-5).all())


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("ker_shape,u_shape", [
    ((1, 1, 1), (1, 3)),
    ((9, 2, 3), (3, 5)),
    ((3, 2, 9), (9, 3)),
    ((4, 15, 3), (3, 3)),
])
def test_push_orth_center_forward(
    subkey: jnp.ndarray,
    ker_shape: Tuple[int, int, int],
    u_shape: Tuple[int, int],
):
    key, subkey = random.split(subkey)
    ker = random_normal_complex(subkey, ker_shape)
    key, subkey = random.split(key)
    u = random_normal_complex(subkey, u_shape)
    key, subkey = random.split(key)
    spec = random.uniform(subkey, (u_shape[1],))
    new_ker, new_u, new_spec = _push_orth_center_forward(ker, u, spec)
    before = jnp.tensordot(ker, u, axes=1) * spec
    after = jnp.tensordot((new_u * new_spec), new_ker, axes=1)
    assert((jnp.abs(before - after) < 1e-5).all())
    kerker = jnp.tensordot(new_ker, new_ker.conj(), axes=[[1, 2], [1, 2]])
    assert((jnp.abs(kerker - jnp.eye(new_ker.shape[0])) < 1e-5).all())
    assert(min(new_u.shape[0], new_ker.shape[1] * new_ker.shape[2]) == new_spec.shape[0])


@pytest.mark.parametrize("spec,eps,true_trsh", [
    (jnp.array([100., 10., 1., 0.1, 0.01, 0.001]), 0.001001 / 100.503781526, 5),
    (jnp.array([100., 10., 1., 0.1, 0.01, 0.001]), 0.01 / 100.503781526,  5),
    (jnp.array([100., 10., 1., 0.1, 0.01, 0.001]), 0.1 / 100.503781526,   4),
    (jnp.array([100., 10., 1., 0.1, 0.01, 0.001]), 1. / 100.503781526,    3),
    (jnp.array([100., 10., 1., 0.1, 0.01, 0.001]), 10. / 100.503781526,   2),
    (jnp.array([100., 10., 1., 0.1, 0.01, 0.001]), 100. / 100.503781526,  1),
    (jnp.array([100., 10., 1., 0.1, 0.01, 0.001]), 1000. / 100.503781526, 0),
])
def test_set_rank(
    spec: jnp.ndarray,
    eps: float,
    true_trsh: int,
):
    trsh = _set_rank(spec, eps)
    assert(trsh == true_trsh)


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("size", [0, 1, 2, 3, 25])
@pytest.mark.parametrize("rank", [1, 5])
def test_truncate_forward_canonical(
    size: int,
    subkey: jnp.ndarray,
    rank: int,
):
    mps = random_normal_mps(subkey, size, rank)
    set_to_forward_canonical(mps)
    mps_copy = deepcopy(mps)
    truncate_forward_canonical(mps, 1_000_000_000)
    log_norm, _ = log_dot(mps_copy, conj(mps))
    assert(jnp.abs(log_norm) < 1e-5)
    log_norm, _ = log_dot(mps, conj(mps))
    assert(jnp.abs(log_norm) < 1e-5)
    for ker in mps:
        kerker = jnp.tensordot(ker, ker.conj(), axes=[[1, 2], [1, 2]])
        assert((jnp.abs(kerker - jnp.eye(kerker.shape[0])) < 1e-5).all())


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("size", [0, 1, 2, 3, 25])
@pytest.mark.parametrize("rank", [1, 5])
def test_truncate_forward_canonical_by_error(
    size: int,
    subkey: jnp.ndarray,
    rank: int,
):
    mps = random_normal_mps(subkey, size, rank)
    set_to_forward_canonical(mps)
    mps_copy = deepcopy(mps)
    truncate_forward_canonical_by_error(mps, 1e-10)
    log_norm, _ = log_dot(mps_copy, conj(mps))
    assert(jnp.abs(log_norm) < 1e-5)
    log_norm, _ = log_dot(mps, conj(mps))
    assert(jnp.abs(log_norm) < 1e-5)
    for ker in mps:
        kerker = jnp.tensordot(ker, ker.conj(), axes=[[1, 2], [1, 2]])
        assert((jnp.abs(kerker - jnp.eye(kerker.shape[0])) < 1e-5).all())


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("size", [0, 1, 2, 3, 25])
@pytest.mark.parametrize("rank", [1, 10])
def test_truncate_mps_by_error(
    size: int,
    subkey: jnp.ndarray,
    rank: int, 
):
    mps = random_normal_mps(subkey, size, rank)
    mps_copy = deepcopy(mps)
    truncate_mps_by_error(mps, 1e-8)
    log_norm1 = set_to_forward_canonical(mps)
    log_norm2 = set_to_forward_canonical(mps_copy)
    assert(jnp.abs(log_norm1 - log_norm2) < 1e-5)
    d = dot(mps, conj(mps_copy))
    assert(jnp.abs(d - 1.) < 1e-5)

@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("size", [0, 1, 2, 3, 25])
@pytest.mark.parametrize("rank", [1, 10])
def test_truncate_mps(
    size: int,
    subkey: jnp.ndarray,
    rank: int, 
):
    mps = random_normal_mps(subkey, size, rank)
    mps_copy = deepcopy(mps)
    truncate_mps(mps, 1_000_000)
    log_norm1 = set_to_forward_canonical(mps)
    log_norm2 = set_to_forward_canonical(mps_copy)
    assert(jnp.abs(log_norm1 - log_norm2) < 1e-5)
    d = dot(mps, conj(mps_copy))
    assert(jnp.abs(d - 1.) < 1e-5)

@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("size", [1, 2, 3, 25])
@pytest.mark.parametrize("rank", [1, 5])
def test_mul_by_log_scalar(
    size: int,
    subkey: jnp.ndarray,
    rank: int,
):
    mps = random_normal_mps(subkey, size, rank)
    set_to_forward_canonical(mps)
    mps_copy = deepcopy(mps)
    mul_by_log_scalar(mps, 12.456)
    log_norm = set_to_forward_canonical(mps)
    print(log_norm)
    assert(jnp.abs(log_norm - 12.456) < 1e-5)
    d = dot(mps, conj(mps_copy))
    assert(jnp.abs(d - 1.) < 1e-5)


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("size", [1, 2, 3, 25])
@pytest.mark.parametrize("rank", [1, 5])
def test_conversions(
    subkey: jnp.ndarray,
    size: int,
    rank: int,
):
    pure_state = random_pure_state(subkey, size, rank)
    assert(len(pure_state) == size)
    d = dot(pure_state, conj(pure_state))
    assert(jnp.abs(d - 1.) < 1e-5)
    dens = pure_state2dens(pure_state)
    trace = dens_trace(dens)
    assert(jnp.abs(trace - 1.) < 1e-5)
    corr = dens2corr(dens)
    assert(len(corr) == size)
    dens_from_corr = corr2dens(corr)
    fid1 = fidelity(dens, pure_state)
    fid2 = fidelity(dens_from_corr, pure_state)
    set_to_forward_canonical(dens_from_corr)
    set_to_forward_canonical(dens)
    dens_dot = dot(dens, conj(dens_from_corr))
    assert(jnp.abs(fid1 - 1.) < 1e-5)
    assert(jnp.abs(fid2 - 1.) < 1e-5)
    assert(jnp.abs(dens_dot - 1.) < 1e-5)


@pytest.mark.parametrize("subkey", random.split(key, 2))
@pytest.mark.parametrize("rank", [1, 5])
def test_eval(
    subkey: jnp.ndarray,
    rank: int,
):
    mps = random_normal_mps(subkey, 7, rank)
    sum_val = sum(mps)
    indices = jnp.arange(0, 2 ** 7)
    indices = jnp.array(jnp.unravel_index(indices, 7 * [2]))
    indices = indices.T
    sum_val_from_eval = eval(mps, indices).sum()
    assert(jnp.abs(sum_val - sum_val_from_eval) < 1e-5)
