import jax.numpy as jnp
from jax import random
from sampler import im2sampler, povm, get_samples
from hdf5_utils import hdf2tt

key = random.PRNGKey(42)

def test_im2sampler():
    samples_num = 100
    im = hdf2tt('hb_it.hdf5')
    def transform(ker):
        lb = ker.shape[0]
        rb = ker.shape[2]
        return ker.reshape((lb, 4, 4, rb))
    im = [transform(ker) for ker in im]
    prob = im2sampler(im, povm)
    random_condition = random.categorical(key, logits=jnp.ones((9,)), shape=(samples_num, 30))
    l = jnp.ones((samples_num, 1))
    log_norms = jnp.zeros((samples_num,))
    for i, ker in enumerate(prob):
        lb = ker.shape[0]
        rb = ker.shape[5]
        ker = ker.sum((1, 3)).reshape((lb, -1, rb))[:, random_condition[:, i]].transpose((1, 0, 2))
        l = jnp.einsum("bi,bij->bj", l, ker)
        norm = jnp.linalg.norm(l, axis=1, keepdims=True)
        l /= norm
        log_norms += jnp.log(norm[:, 0])
    assert(jnp.all((jnp.abs((log_norms - log_norms[0])) < 1e-4)))


def test_get_samples():
    indices = random.categorical(random.PRNGKey(42), jnp.ones((3,)), shape=(1000, 60))
    im = hdf2tt('hb_it.hdf5')
    def transform(ker):
        lb = ker.shape[0]
        rb = ker.shape[2]
        return ker.reshape((lb, 4, 4, rb))
    im = [transform(ker) for ker in im]
    sampler = im2sampler(im, povm)
    for _ in range(10):
        print(get_samples(random.PRNGKey(42), sampler, indices).shape)

test_get_samples()