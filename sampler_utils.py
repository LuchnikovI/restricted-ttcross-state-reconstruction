from functools import partial
import jax.numpy as jnp
from jax import vmap, jit, random
from constants import xyz
from typing import List


_, v = jnp.linalg.eigh(xyz)
povm = v[:, :, jnp.newaxis] * v[:, jnp.newaxis].conj()
povm = povm.transpose((3, 2, 1, 0))
povm = povm.reshape((2, 4, 3))


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

        povm: an conditional POVM of shape [2, 4, observables_number], here is the pictorial(TN)
            representation:

                         rho
                         (1)
                          |
                      #########
       outcome (0) ---#########--- (2) observable id
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
        lb, _, rb = ker.shape
        ker = ker.reshape((lb, 4, 4, rb))
        ker = ker / 2.
        ker = jnp.tensordot(ker, povm, axes=[[1], [1]])
        ker = jnp.tensordot(ker, povm, axes=[[1], [1]])
        ker = ker.transpose((0, 2, 3, 4, 5, 1))
        return ker
    return [transform(ker) for ker in im]


@jit
@partial(vmap, in_axes=(None, 0, 0))
def push_plug_to_left(
    ker: jnp.ndarray,
    prev_plug: jnp.ndarray,
    indices: jnp.ndarray
) -> jnp.ndarray:
    ker = ker[:, :, indices[0], :, indices[1]]
    ker = ker.sum((1, 2))
    new_plug = jnp.tensordot(ker, prev_plug, axes=1)
    new_plug /= jnp.linalg.norm(new_plug)
    return new_plug


@jit
@partial(vmap, in_axes=(None, 0, 0, 0))
def push_plug_to_right(
    ker: jnp.ndarray,
    prev_plug: jnp.ndarray,
    indices: jnp.ndarray,
    samples: jnp.ndarray,
) -> jnp.ndarray:
    ker = ker[:, samples[0], indices[0], samples[1], indices[1]]
    new_plug = jnp.tensordot(prev_plug, ker, axes=1)
    new_plug /= jnp.linalg.norm(new_plug)
    return new_plug


@jit
@partial(vmap, in_axes=(0, None, 0, 0))
def get_log_prob(
    left_plug: jnp.ndarray,
    ker: jnp.ndarray,
    right_plug: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    ker = ker[:, :, indices[0], :, indices[1]]
    prob = jnp.tensordot(left_plug, ker, axes=1)
    prob = jnp.tensordot(prob, right_plug, axes=1)
    return jnp.log(prob)

@jit
@partial(vmap, in_axes=(0, 0))
def _sample_from_prob(
    gumbel_sample: jnp.ndarray,
    log_prob: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.array(jnp.unravel_index(jnp.argmax(gumbel_sample + log_prob), log_prob.shape))

@jit
def sample_from_prob(
    key: jnp.ndarray,
    log_prob: jnp.ndarray,
) -> jnp.ndarray:
    gumbel_sample = random.gumbel(key, log_prob.shape)
    return _sample_from_prob(gumbel_sample, log_prob)


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
