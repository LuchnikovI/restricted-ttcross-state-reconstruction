import sys
import json
import matplotlib.pyplot as plt
import numpy as np

def main(argv):
    with open(argv[1], "r") as results:
        data = json.load(results)
    corr_cosine_sims = np.array([float(v) for v in data["corr_cosine_sim"]])
    recon_uncontrolled_prediction = np.array([[float(v) for v in bloch_vec] for bloch_vec in data["recon_uncontrolled_prediction"][-1]])
    recon_controlled_prediction = np.array([[float(v) for v in bloch_vec] for bloch_vec in data["recon_controlled_prediction"][-1]])
    uncontrolled_prediction = np.array([[float(v) for v in bloch_vec] for bloch_vec in data["uncontrolled_prediction"][-1]])
    controlled_prediction = np.array([[float(v) for v in bloch_vec] for bloch_vec in data["controlled_prediction"][-1]])
    trace = np.array([float(v) for v in data["trace"]])
    config = data["config"]
    noise_std = config["noise_std"]
    restrictions = np.array([int(v) for v in config["restrictions"]])
    plt.figure()
    plt.plot(restrictions, len(restrictions) * [1.], '-or')
    plt.plot(restrictions, corr_cosine_sims, '-ob')
    plt.legend(["What we need.", "What we get."])
    plt.title("noise_std: {}".format(noise_std))
    plt.yscale("log")
    plt.ylabel("Cosine similarity")
    plt.xlabel("At most number of corr. fn points.")
    plt.savefig("cosine_sim.pdf")
    plt.figure()
    plt.plot(restrictions, trace, '-ob')
    plt.title("noise_std: {}".format(noise_std))
    plt.ylim(0, 1.1)
    plt.ylabel("Trace")
    plt.xlabel("At most number of corr. fn points.")
    plt.savefig("trace.pdf")
    plt.figure()
    plt.plot(uncontrolled_prediction[:, 0], "-ob")
    plt.plot(recon_uncontrolled_prediction[:, 0], ":dr")
    plt.ylabel("<X>")
    plt.ylim(top=1, bottom=-1)
    plt.xlabel("Time")
    plt.savefig("uncontrolled_prediction.pdf")
    plt.figure()
    plt.plot(controlled_prediction[:, 0], "-ob")
    plt.plot(recon_controlled_prediction[:, 0], ":dr")
    plt.ylabel("<X>")
    plt.ylim(top=1, bottom=-1)
    plt.xlabel("Time")
    plt.savefig("controlled_prediction.pdf")

if __name__ == '__main__':
    main(sys.argv)