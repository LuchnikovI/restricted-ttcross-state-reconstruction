import sys
import jax.numpy as jnp
from jax import random
import numpy as np
from tqdm import tqdm
import json
from ttrs import TTVc64

import sys
sys.path.append("../")
from mps_utils import *
from im_utils import *
from externel_im_utils import *

from jax.config import config
config.update("jax_enable_x64", True)

max_n = 0  # this global var. keeps track of maximal points of a corr func during a session

def main(argv):
    config_path = argv[1]
    with open(config_path, "r") as config:
        data = json.load(config)
    restrictions = [int(restriction / 2) for restriction in data["restrictions"]]
    key = random.PRNGKey(int(data["prngkey"]))
    ttcross_rank = data["ttcross_rank"]
    sweeps_num = data["sweeps_num"]
    noise_std = data["noise_std"]
    path = data["path"]
    im = hdf2tt(path)
    time_steps = len(im)
    #rank = 0
    #for ker in im:
    #    lb, _, rb = ker.shape()
    #    rank = max(lb, rank)
    #    rank = max(rb, rank)
    result = {
        "config": None,
        "corr_cosine_sim": [],
        "recon_uncontrolled_prediction": [],
        "uncontrolled_prediction": [],
        "recon_controlled_prediction": [],
        "controlled_prediction": [],
        "bond_dims_of_truncated_im": [],
        "controlled_id_im_prediction": [],
    }
    for restriction in restrictions:
        print("Experiment for at most {}-points corr. function has started.".format(2 * restriction))
        print("// ------------------------------------------------------ //")
        result["bond_dims_of_truncated_im"].append(
            [ker.shape[-1] for ker in im][:-1]
        )
        corr = im2corr(im)
        # ttcross initialization
        tt = TTVc64(
            time_steps * [16], # modes dimensions of a tensor train
            ttcross_rank,         # max TT rank
            0.,                   # accuracy of the maxvol algorithm (here makes no effect)
            False                 # flag showing if we need to track data for TTOpt method (https://arxiv.org/abs/2205.00293)
        )
        tt.restrict(restriction)
        # counter reset
        global max_n
        max_n = 0
        # this function is being called by ttcross
        def fun(x: jnp.ndarray) -> jnp.ndarray:
            global max_n
            max_n = max(max_n, check_indices(x))
            vals = eval(corr, x)
            vals =  vals + noise_std * np.random.normal(vals.shape)
            return vals
        # main reconstruction loop
        for _ in tqdm(range(time_steps * sweeps_num)):
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
        #set_to_forward_canonical(recon_corr)
        #truncate_forward_canonical(recon_corr, truncation_rank)
        recon_im = corr2im(recon_corr)
        recon_uncontrolled_prediction = list(map(dens2bloch, get_dynamics(time_steps * [jnp.eye(4)], recon_im, jnp.array([1., 0., 0., 0.]))))
        uncontrolled_prediction = list(map(dens2bloch, get_dynamics(time_steps * [jnp.eye(4)], im, jnp.array([1., 0., 0., 0.]))))
        control = get_random_unitary_phi(key, time_steps)
        recon_controlled_prediction = list(map(dens2bloch, get_dynamics(control, recon_im, jnp.array([1., 0., 0., 0.]))))
        controlled_prediction = list(map(dens2bloch, get_dynamics(control, im, jnp.array([1., 0., 0., 0.]))))
        controlled_id_im_prediction = list(map(dens2bloch, get_dynamics(control, get_id_im(time_steps), jnp.array([1., 0., 0., 0.]))))
        result["recon_uncontrolled_prediction"].append([np.array(dens).tolist() for dens in recon_uncontrolled_prediction])
        result["uncontrolled_prediction"].append([np.array(dens).tolist() for dens in uncontrolled_prediction])
        result["recon_controlled_prediction"].append([np.array(dens).tolist() for dens in recon_controlled_prediction])
        result["controlled_prediction"].append([np.array(dens).tolist() for dens in controlled_prediction])
        result["controlled_id_im_prediction"].append([np.array(dens).tolist() for dens in controlled_id_im_prediction])
        set_to_forward_canonical(corr)
        set_to_forward_canonical(recon_corr)
        corr_cosine_sim = dot(corr, conj(recon_corr)).real
        result["corr_cosine_sim"].append(str(float(corr_cosine_sim)))
    result["config"] = data
    with open("result.json", "w") as outfile:
        outfile.write(json.dumps(result))

if __name__ == '__main__':
    main(sys.argv)
