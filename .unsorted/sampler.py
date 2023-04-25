from typing import List, Tuple
import jax.numpy as jnp
from jax import jit, random
from sampler_utils import (
    push_plug_to_left,
    push_plug_to_right,
    get_log_prob,
    sample_from_prob,
)

# Pauli
sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
s0 = jnp.eye(2, dtype=jnp.complex64)
xyz = jnp.concatenate([sx[jnp.newaxis], sy[jnp.newaxis], sz[jnp.newaxis]], axis=0)

# Cond. POVM
_, v = jnp.linalg.eigh(xyz)
povm = v[:, :, jnp.newaxis] * v[:, jnp.newaxis].conj()
povm = povm.transpose((3, 2, 1, 0))
povm = povm.reshape((2, 4, 3))

@jit
def im2sampler(
    im: List[jnp.ndarray],
    povm:  jnp.ndarray,
) -> List[jnp.ndarray]:
    '''Transforms IM to an MPS based sampler.
    Args:
        im: list of kernels, each kernel has shape [lb, 4, 4, rb],
            here is the pictorial(TN) representation of a kernel:

                        ###########
        left_bond (0)---###########--- (3)right_bond
                        ###########
                          |     |
                         (1)   (2)
                        rho1   rho2

        povm: an conditional POVM of shape [2, 4, 4], here is the pictorial(TN)
            representation:

                         rho
                         (1)
                          |
                      #########
       outcome (0) ---#########--- (2) obs_num
                      #########
    Returns:
        mps like decomposition representing conditional probability distribution for
        sampling measurement outcomes, here is the pictorial(TN) representation of
        an MPS kernel:

                        ###########
        left_bond (0)---###########--- (3)right_bond
                        ###########
                        |  |   |  |
                       (1)(2) (3)(4)
                        o  o   o  o
                        u  b   u  b
                        t  s   t  s
                        c      c
                        o  n   o  n
                        m  u   m  u
                        e  m   e  m

    '''

    def transform(ker: jnp.ndarray) -> jnp.ndarray:
        ker = jnp.tensordot(ker, povm, axes=[[1], [1]])
        ker = jnp.tensordot(ker, povm, axes=[[1], [1]])
        ker = ker.transpose((0, 2, 3, 4, 5, 1))
        return ker
    return [transform(ker) for ker in im]

@jit
def get_samples(
    key: jnp.ndarray,
    tt_prob: List[jnp.ndarray],
    indices: jnp.ndarray,
) -> jnp.ndarray:
    samples_num = indices.shape[0]
    kers_num = len(tt_prob)
    right_plugs = [jnp.ones((samples_num, 1))]
    indices = indices.reshape((samples_num, -1, 2))
    for i in reversed(range(0, kers_num)):
        ker = tt_prob[i]
        index = indices[:, i]
        new_plug = push_plug_to_left(ker, right_plugs[0], index)
        right_plugs = [new_plug] + right_plugs
    left_plug = jnp.ones((samples_num, 1))
    samples = jnp.ones((samples_num, 0))
    for index, ker, right_plug in zip(indices.transpose((1, 0, 2)), tt_prob, right_plugs[1:]):
        key, subkey = random.split(key)
        log_prob = get_log_prob(left_plug, ker, right_plug, index)
        sample = sample_from_prob(subkey, log_prob.real)
        samples = jnp.concatenate([samples, sample], axis=1)
        left_plug = push_plug_to_right(ker, left_plug, index, sample)
    return samples
        