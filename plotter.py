import sys
import json
import matplotlib.pyplot as plt
import numpy as np

def main(argv):
    with open(argv[1], "r") as results:
        data = json.load(results)
    cosine_sims = np.array([float(v) for v in data["cosine_sim"]])
    max_ns = np.array([int(v) for v in data["max_n"]])
    mean_errs = np.array([float(v) for v in data["mean_err"]])
    config = data["config"]
    restrictions = np.array([int(v) for v in config["restrictions"]])
    qubits_number = config["qubits_number"]
    max_mps_rank = config["max_mps_rank"]
    ttcross_rank = config["ttcross_rank"]
    sweeps_num = config["sweeps_num"]
    random_indices_num = config["random_indices_num"]
    plt.figure()
    plt.plot(restrictions, cosine_sims, '-ob')
    plt.title("qubits_num: {}, dens_matrix_rank: {}".format(qubits_number, max_mps_rank ** 2))
    plt.ylabel("Cosine similarity")
    plt.xlabel("At most number of corr. fn points.")
    plt.savefig("cosine_sim.pdf")
    plt.figure()
    plt.plot(restrictions, mean_errs, '-ob')
    plt.title("qubits_num: {}, dens_matrix_rank: {}".format(qubits_number, max_mps_rank ** 2))
    plt.ylabel("Error per {} samples".format(random_indices_num))
    plt.xlabel("At most number of corr. fn points.")
    plt.yscale('log')
    plt.savefig("err.pdf")



if __name__ == '__main__':
    main(sys.argv)