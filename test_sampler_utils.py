from sampler_utils import *
from im_utils import *
from jax import random

key = random.PRNGKey(42)

def test_povm():
    for p in povm.transpose((2, 0, 1)):
        p = p.reshape((2, 2, 2))
        assert((jnp.abs(p[0].T.conj() - p[0]) < 1e-5).all())
        assert((jnp.abs(p[1].T.conj() - p[1]) < 1e-5).all())
        assert((jnp.abs(p[0] @ p[0] - p[0]) < 1e-5).all())
        assert((jnp.abs(p[1] @ p[1] - p[1]) < 1e-5).all())
        assert((jnp.abs(p[1] @ p[0]) < 1e-5).all())

def test_sampler_placegolder():
    im = get_random_im(key, 10, 2, 20)
    sampler = im2sampler(im, povm)
    _, subkey1, subkey2 = random.split(key, 3)
    samples = get_samples(subkey1, sampler, random.categorical(subkey2, jnp.ones((4)), shape = (1000, 20)))
    print(samples.shape)

test_sampler_placegolder()