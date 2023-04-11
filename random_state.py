import sys
import jax.numpy as jnp
from jax import random
import numpy as np
from tqdm import tqdm
import json
from mps_utils import *
from ttrs import TTVc64

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
    random_indices_num = data["random_indices_num"]
    result = {"config": None, "max_n": [], "cosine_sim": [], "mean_err": []}
    for restriction in restrictions:
        print("Experiment for at most {}-points corr. function has started.".format(restriction))
        print("// ------------------------------------------------------ //")
        corr = random_corr(key, qubits_number, max_mps_rank)  # corr. function has rank max_mps_rank ** 2
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
            return jnp.exp(log_ampl) * sign
        # main reconstruction loop
        for i in tqdm(range(qubits_number * sweeps_num)):
            index = tt.get_args()
            if index is None:
                tt.update(None)
            else:
                val = fun(index)
                tt.update(np.array(val))
        # processing of results
        result["max_n"].append(str(max_n))
        recon_corr = tt.get_kernels()
        key, subkey = random.split(key)
        random_indices = random.categorical(subkey, jnp.array([1., 1., 1., 1.]), shape = (random_indices_num, qubits_number))
        log_corr_val, _ = log_eval(corr, random_indices)
        log_corr_val_recon, _ = log_eval(recon_corr, random_indices)
        mean_err = jnp.abs((log_corr_val - log_corr_val_recon) / (log_corr_val + log_corr_val_recon)).mean()
        result["mean_err"].append(str(mean_err))
        set_to_forward_canonical(recon_corr)
        set_to_forward_canonical(corr)
        cosine_sim = jnp.real(dot(recon_corr, corr))
        result["cosine_sim"].append(str(float(cosine_sim)))
    result["config"] = data
    with open("result.json", "w") as outfile:
        outfile.write(json.dumps(result))

if __name__ == '__main__':
    main(sys.argv)
