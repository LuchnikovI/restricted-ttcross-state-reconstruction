from jax import random
import jax.numpy as jnp
from mps_utils import (
    set_to_forward_canonical,
    random_pure_state,
    fidelity,
    pure_state2dens,
    dens2corr,
    corr2dens,
    conj,
    dot,
    dens_trace,
)

def test_conversions():
    key = random.PRNGKey(42)
    pure_state = random_pure_state(key, 60, 20)
    assert(len(pure_state) == 60)
    dot_val = dot(pure_state, conj(pure_state))
    dens = pure_state2dens(pure_state)
    trace1 = dens_trace(dens)
    assert(len(dens) == 60)
    assert(jnp.abs(trace1 - 1.) < 1e-5)
    corr = dens2corr(dens)
    assert(len(corr) == 60)
    dens_from_corr = corr2dens(corr)
    trace2 = dens_trace(dens_from_corr)
    assert(len(dens_from_corr) == 60)
    assert(jnp.abs(trace2 - 1.) < 1e-5)
    fid1 = fidelity(dens, pure_state)
    fid2 = fidelity(dens_from_corr, pure_state)
    set_to_forward_canonical(dens_from_corr)
    set_to_forward_canonical(dens)
    dens_dot = dot(dens, conj(dens_from_corr))
    
    assert(jnp.abs(dot_val - 1.) < 1e-5)
    assert(jnp.abs(fid1 - 1.) < 1e-5)
    assert(jnp.abs(fid2 - 1.) < 1e-5)
    assert(jnp.abs(dens_dot -1.) < 1e-5)
