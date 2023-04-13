## What this experiment does?

- One generates a random pure state in the MPS form, builds MPO density matrix from this MPS, applies a complete set of local observables to each pair of indices of the MPO density matrix corresponding to a single subsystem to get a tensor representing all possible correlation functions. Here is the corresponding illustration of this tensor built out of local observables and MPS, we call this tensor a correlation tensor:
```  
      @@@      @@@      @@@
...===@@@======@@@======@@@===...  <------- MPS
      @@@      @@@      @@@
       |        |        |
       |        |        |
       |   @@@  |   @@@  |   @@@
  ...======@@@======@@@======@@@===...  <------- conjugated MPS
       |   @@@  |   @@@  |   @@@
       |    |   |    |   |    |
       |    |   |    |   |    |
        \  /     \  /     \  /
         @@       @@       @@  <------- local observables, tensors consisting of sigma_0, sigma_x, sigmz_y, sigma_z
         @@       @@       @@           value 0 of a dangling index corresponds to sigma_0, 1, 2, 3 correspond
         |        |        |            to sigma_x, sigma_y, sigma_z respectively
         |        |        |
```
- Runs the restricted TTCross algorithm that calls noisy elements of the correlation tensor by index and tries to reconstruct the correlation tensor from this information. Note, that you explicitly restrict the set of elements that are allowed to be called by setting the maximum number of non-zero indices in each element of the correlation tensor. If you restrict this number by $n$ it corresponds to the case when you have access to only values of $<n-$ points correlation functions. Therefore, this method is well combined with recent shadow tomography technique that allows one to reconstruct precisely this information from measurements;
- Reconstructs the density matrix from the reconstructed correlation tensor;
- To validate reconstruction results, it saves the following data in the `result.json` file:
  1) cosine similarity between actual correlation tensor and its reconstruction;
  2) infidelity between reconstructed state and exact state;
  3) trace of the reconstructed state.

## How to run experiments

- One needs to specify parameters of the experiment in the `config.json` file. Here is the sample of this file with some explanation of its fields:
```
{
    "ttcross_rank": 30,            # maximal rank used by the restricted TTCross algorithm during reconstruction

    "sweeps_num": 6,               # number of dmrg-sweeps used by the reconstruction algorithm 

    "restrictions": [3, 5, 7, 9],  # set of restrictions (max. number of points in a corr. function). 
                                   # For each value one runs a separate experiment.

    "qubits_number": 60,           # qubits number in a random state

    "max_mps_rank": 5,             # max rank of the state. The corresponding density matrix has rank max_mps_rank^2.
                                   # Note, that to guaranty reliable reconstruction ttcross_rank must be >= max_mps_rank^2.

    "noise_std": 0.001             # std of additive i.i.d. gaussian noise that is added to measured values

    "prngkey": 42,                 # A random seed, any positive integer
}
```
- One runs experiments as follows `python random_state.py config.json`.

## How to plot results

To plot the obtained results saved in `result.json` file, one runs `python plotter.py results.json`. This command generates corresponding plots in `.pdf` format and saves them in the same directory.