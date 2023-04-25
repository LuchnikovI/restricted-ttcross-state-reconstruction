from typing import List
import jax.numpy as jnp
import re
import h5py


def hdf2tt(path: str) -> List[jnp.ndarray]:
    tt_data = h5py.File(path, 'r')
    kers = tt_data['tmax=60'].values()
    result = (len(kers) - 2) * [None]
    for ker in kers:
        idx = re.findall(r'M_(\d+)', ker.name)
        if len(idx) == 0:
            continue
        result[int(idx[0])] = jnp.array(ker)
    return result