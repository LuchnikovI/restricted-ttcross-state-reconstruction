from sampler_utils import push_plug_to_left, _sample_from_prob
import jax.numpy as jnp
from jax import random

def test_plush_plug_left():
    key, subkey = random.split(random.PRNGKey(42))
    prev_plug = random.normal(key, (100, 5))
    key, subkey = random.split(subkey)
    ker = random.normal(key, (5, 2, 3, 2, 3, 5))
    key, _ = random.split(subkey)
    indices = random.categorical(key, jnp.ones((3,)), shape=(100, 2))
    subplug = push_plug_to_left(ker, prev_plug, indices)[15]
    index = indices[15]
    true_subplug = jnp.tensordot(ker[:, :, index[0], :, index[1]].sum((1, 2)), prev_plug[15], axes=1)
    assert(jnp.all(jnp.abs(true_subplug - subplug) < 1e-6))

def test__sample_from_prob():
    fake_gumbel = jnp.array([
        [[1, 2, 5], [3, 7, 6]],
        [[-1, -1, -1], [-1, -1, -1]],
    ], dtype=jnp.float32)
    fake_log_prob = jnp.array([
        [[-1, -1, -1], [-1, -1, -1]],
        [[2, 2, 3], [4, 3, 5]],
    ], dtype=jnp.float32)
    assert(jnp.all(_sample_from_prob(fake_gumbel, fake_log_prob) == jnp.array([[1, 1], [1, 2]])))
