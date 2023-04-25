from typing import List
import jax.numpy as jnp
import re
import h5py


def hdf2tt(path: str) -> List[jnp.ndarray]:
    tt_data = h5py.File(path, 'r')
    key = [key for key in tt_data.keys()][0]
    kers = tt_data[key].values()
    result = len(kers) * [None]
    for ker in kers:
        idx = re.findall(r'M_(\d+)', ker.name)
        if len(idx) == 1:
            result[int(idx[0])] = jnp.array(ker)
    def f(x):
        if x is None:
            return False
        elif x.shape == (1, 1, 1):
            return False
        elif x.shape == ():
            return False
        else:
            return True
    result = list(filter(f, result))
    result.reverse()
    for i in range(len(result)):
        ker = result[i]
        lb, _, rb = ker.shape
        ker = ker.reshape((lb, 4, 4, rb))
        ker = ker.transpose((3, 2, 1, 0))
        ker = ker.reshape((rb, 16, lb))
        result[i] = ker
    return result
