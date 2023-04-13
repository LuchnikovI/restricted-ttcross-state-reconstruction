import sys
import json
import matplotlib.pyplot as plt
import numpy as np

def main(argv):
    with open(argv[1], "r") as results:
        data = json.load(results)
    corr_cosine_sims = np.array([float(v) for v in data["corr_cosine_sim"]])
    infid = np.array([float(v) for v in data["infid"]])
    trace = np.array([float(v) for v in data["trace"]])
    config = data["config"]
    noise_std = config["noise_std"]
    restrictions = np.array([int(v) for v in config["restrictions"]])
    qubits_number = config["qubits_number"]
    max_mps_rank = config["max_mps_rank"]
    plt.figure()
    plt.plot(restrictions, len(restrictions) * [1.], '-or')
    plt.plot(restrictions, corr_cosine_sims, '-ob')
    plt.legend(["What we need.", "What we get."])
    plt.title("qubits_num: {}, dens_matrix_rank: {}, noise_std: {}".format(qubits_number, max_mps_rank ** 2, noise_std))
    plt.yscale("log")
    plt.ylabel("Cosine similarity")
    plt.xlabel("At most number of corr. fn points.")
    plt.savefig("cosine_sim.pdf")
    plt.figure()
    plt.plot(restrictions, infid, '-ob')
    plt.title("qubits_num: {}, dens_matrix_rank: {}, noise_std: {}".format(qubits_number, max_mps_rank ** 2, noise_std))
    plt.yscale("log")
    plt.ylabel("Infidelity")
    plt.xlabel("At most number of corr. fn points.")
    plt.savefig("infidelity.pdf")
    plt.figure()
    plt.figure()
    plt.plot(restrictions, trace, '-ob')
    plt.title("qubits_num: {}, dens_matrix_rank: {}, noise_std: {}".format(qubits_number, max_mps_rank ** 2, noise_std))
    plt.ylim(0, 1.1)
    plt.ylabel("Trace")
    plt.xlabel("At most number of corr. fn points.")
    plt.savefig("trace.pdf")

if __name__ == '__main__':
    main(sys.argv)