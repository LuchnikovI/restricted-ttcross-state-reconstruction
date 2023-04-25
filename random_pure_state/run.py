import sys
import jax.numpy as jnp
from jax import random
import numpy as np
from tqdm import tqdm
import json
from ttrs import TTVc64

import sys;
sys.path.append("../") 
from mps_utils import *

from jax.config import config
config.update("jax_enable_x64", True)

max_n = 0  # this global var. keeps track of maximal points of a corr func during a session

def main(argv):
    config_path = argv[1]
    with open(config_path, "r") as config:
        data = json.load(config)
    restrictions = data["restrictions"]
    key = random.PRNGKey(int(data["prngkey"]))
    qubits_number = data["qubits_number"]
    max_mps_rank = data["max_mps_rank"]
    ttcross_rank = data["ttcross_rank"]
    sweeps_num = data["sweeps_num"]
    noise_std = data["noise_std"]
    result = {
        "config": None,
        "corr_cosine_sim": [],
        "infid": [],
        "trace": [],
    }
    for restriction in restrictions:
        print("Experiment for at most {}-points corr. function has started.".format(restriction))
        print("// ------------------------------------------------------ //")
        psi = random_pure_state(key, qubits_number, max_mps_rank)
        dens = pure_state2dens(psi)
        corr = dens2corr(dens)
        # ttcross initialization
        tt = TTVc64(
            qubits_number * [4], # modes dimensions of a tensor train
            ttcross_rank,        # max TT rank
            0.,                  # accuracy of the maxvol algorithm (here makes no effect)
            False                # flag showing if we need to track data for TTOpt method (https://arxiv.org/abs/2205.00293)
        )
        tt.restrict(restriction)
        # counter reset
        global max_n
        max_n = 0
        # this function is being called by ttcross
        def fun(x: jnp.ndarray) -> jnp.ndarray:
            global max_n
            max_n = max(max_n, check_indices(x))
            log_ampl, sign = log_eval(corr, x)
            return jnp.exp(log_ampl) * sign + noise_std * np.random.normal(log_ampl.shape)
        # main reconstruction loop
        for i in tqdm(range(qubits_number * sweeps_num)):
            index = tt.get_args()
            if index is None:
                tt.update(None)
            else:
                val = fun(index)
                tt.update(np.array(val))
        # processing of results
        if max_n > restriction:
            raise RuntimeError("A reconstruction process called a forbidden corr.function.")
        recon_corr = tt.get_kernels()
        recon_dens = corr2dens(recon_corr)
        trace = dens_trace(recon_dens).real
        set_to_forward_canonical(corr)
        set_to_forward_canonical(recon_corr)
        corr_cosine_sim = dot(corr, conj(recon_corr)).real
        infid = 1. - fidelity(recon_dens, psi)
        result["infid"].append(str(float(infid)))
        result["corr_cosine_sim"].append(str(float(corr_cosine_sim)))
        result["trace"].append(str(float(trace)))
    result["config"] = data
    with open("result.json", "w") as outfile:
        outfile.write(json.dumps(result))

if __name__ == '__main__':
    main(sys.argv)
