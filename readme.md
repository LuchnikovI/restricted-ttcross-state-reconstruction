## How to prepare your system for running experiments

You need to configure a python venv and install all the necessary dependencies:
   - run `python -m venv <name>`, where `<name>` is the name of your venv, to create a venv in your working directory;
   - run `pip install maturin` within the created venv to instal maturin -- an app for building python packages from
   rust source code;
   - clone a GitHub repo https://github.com/LuchnikovI/ttrs with ttrs package containing a restricted TTCross algorithm;
   - enter the cloned project's root directory and run `maturin develop --release` within the venv. It will install ttrs as a python package inside your venv;
   - pip install the following packages in your venv: numpy, matplotlib, jax (the cpu version should be enough), tqdm;