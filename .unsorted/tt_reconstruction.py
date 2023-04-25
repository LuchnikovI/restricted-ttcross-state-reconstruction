import numpy as np
import jax.numpy as jnp
from jax import jit
from typing import List, Tuple
from ttrs import TTVc64
import h5py
import re

from jax.config import config
config.update("jax_enable_x64", True)

MODES_NUM = 30

# this one works only for a particular instance of IM
def hdf2tt(path: str) -> List[np.ndarray]:
    tt_data = h5py.File(path, 'r')
    kers = tt_data['tmax=60'].values()
    result = (len(kers) - 2) * [None]
    for ker in kers:
        idx = re.findall(r'M_(\d+)', ker.name)
        if len(idx) == 0:
            continue
        result[int(idx[0])] = np.array(ker)
    return result

@jit
def log_dot(lhs: List[jnp.ndarray], rhs: List[jnp.ndarray]) -> Tuple[jnp.array, jnp.array]:
    log_norm = 0
    l = np.ones((1, 1))
    for (lhs_ker, rhs_ker) in zip(lhs, rhs):
        l = jnp.tensordot(l, lhs_ker, axes=1)
        l = jnp.tensordot(rhs_ker, l, axes=[[0, 1], [0, 1]])
        norm = jnp.linalg.norm(l)
        l /= norm
        log_norm += jnp.log(norm)
    return log_norm, l[0, 0]

@jit
def conj(tt: List[jnp.ndarray]) -> List[jnp.ndarray]:
    tt_conj = []
    for ker in tt:
        tt_conj.append(ker.conj())
    return tt_conj

@jit
def eval(tt: List[jnp.ndarray], indices: jnp.ndarray) -> jnp.array:
    log_norms = jnp.zeros((indices.shape[0],))
    l = jnp.ones((indices.shape[0], 1))
    for (slice, ker) in zip(indices.T, tt):
        l = jnp.einsum("bi,bij->bj", l, ker[:, slice, :].transpose((1, 0, 2)))
        norm = jnp.linalg.norm(l, axis=1, keepdims=True)
        l /= norm
        log_norms += jnp.log(norm[:, 0])
    return jnp.exp(log_norms) * l[:, 0]

tt = TTVc64(
    MODES_NUM * [16], # modes dimensions of a tensor train
    65,              # max TT rank
    1e-5,             # accuracy of the maxvol algorithm
    False             # flag showing if we need to track data for TTOpt method (https://arxiv.org/abs/2205.00293)
)

tt.from_kernels(hdf2tt('hb_it.hdf5'))
tt.set_into_left_canonical()
tt.truncate_left_canonical(1e-1)
raw_tt = tt.get_kernels()

fun = lambda x: eval(raw_tt, x)

ttcross = TTVc64(
    MODES_NUM * [16], # modes dimensions of a tensor train
    65,              # max TT rank
    1e-5,             # accuracy of the maxvol algorithm
    False             # flag showing if we need to track data for TTOpt method (https://arxiv.org/abs/2205.00293)
)

evals_num = 0
for i in range(MODES_NUM * 6):
    print("Iter num. {} done.".format(i))
    index = ttcross.get_args()
    if index is None:
        ttcross.update(None)
    else:
        val = fun(index)
        evals_num += index.shape[0]
        ttcross.update(np.array(val))

tt.set_into_left_canonical()
ttcross.set_into_left_canonical()
ttcross.conj()

log_norm, sign = tt.log_dot(ttcross)

print("Log(norm) for scalar product {}.".format(log_norm))
print("Phase multiplier {}.".format(sign))
