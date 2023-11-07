# Fully differentiable Cellular Potts Model powered by JAX

The cellular Potts model (CPM) recently has seen a lot of attention for the
simulation of cellular dynamics, for instance in the context of tumors.
Traditionally, these types of simulations are hard to incorporate into machine
learning processes due to their stochastic and discrete nature. With the
advances in automatic differentiation however, this aims to be a minimum viable
example of how to implement a fully differentiable Markov Chain Monte Carlo CPM
simulation, ready to use for end-to-end AI training.

## Usage

### Dependencies

This project depends on `jax` for differentiable programming and `matplotlib`
for the visualisation. It contains a `requirements.txt` file to use on machines
with NVIDIA GPUs. On other machines, the dependencies can be installed manually.

#### For use with NVIDIA GPUs:

```bash
pip install -r requirements.txt
```

**NB:** This project does not make use of the GPU at the moment, but has still
been built and tested with GPU support enabled in order to be flexible for later use.

#### Manual

1. Install `jax` by following the [installation manual](https://jax.readthedocs.io/en/latest/installation.html)
2. Install matplotlib (e. g. `pip install matplotlib`)


### Execution

Invoke

```bash
python main.py
```

from the project's root.
