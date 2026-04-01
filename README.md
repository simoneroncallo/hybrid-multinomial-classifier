# Hybrid multinomiall classifier

This repository contains the simulation code for a hybrid multinomial classifier which consists of multiple quantum binary models, combined using post-processing techniques such as one-vs-one, one-vs-rest and a binary decision tree. As a quantum model, we consider a quantum optical shallow network based on the Hong-Ou-Mandel effect. Previously implemented from scratch (see [quantum-optical-network](https://github.com/simoneroncallo/quantum-optical-network)), here we consider a full [PyTorch](https://github.com/pytorch/pytorch) implementation, that leverages its mathematical equivalence with a shallow neural network (with neurons subject to the L2 and L1 normalization constraints). 

Contributors: Angela Rosy Morgillo [@MorgilloR](https://github.com/MorgilloR) and Simone Roncallo [@simoneroncallo](https://github.com/simoneroncallo) <br>
Reference: In Preparation (2026)

## Installation
The Python environment can be configured in rootless [Docker](https://docs.docker.com/) container, by running the script `scripts/build.sh` or
```bash
sudo docker build -t multinomial-classifier .
./scripts/jupyterlab.sh
```
The simulation code is contained in `main.py`. See `tests/` for further investigations.

## Structure
The repository has the following structure
```bash
mltclass
   ├── classical.py # Neural network (PyTorch)
   ├── neuron.py # Quantum optical neuron (PyTorch)
   ├── shallow.py # Quantum optical shallow network (PyTorch)
   └── utils
       ├── dataset.py # Dataset preparation
       ├── metrics.py # Classification performance metrics
       ├── tree.py # Decision tree generation and evaluation
       └── visualize.py 

scripts
   ├── build.sh # Build the Docker image
   ├── getdata.py # Download the dataset
   ├── jupyterlab.sh # Run the container with Jupyter
   └── run.sh

main.py # Simulation code
requirements.txt # Python dependencies
```
