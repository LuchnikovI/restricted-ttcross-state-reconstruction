import jax.numpy as jnp
from externel_im_utils import hdf2tt
from im_utils import im2corr
from mps_utils import eval

#def test_hdf2tt():
#    im = hdf2tt("const_specdens_chi_50.hdf5")
#    corr = im2corr(im)
#    val = eval(corr, jnp.zeros((1, len(corr)), dtype=jnp.uint))
#    assert(jnp.abs(val - 1.) < 1e-5)
