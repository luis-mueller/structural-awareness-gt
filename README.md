# Designing a structural awareness task for graph transformers

[![pytorch](https://img.shields.io/badge/PyTorch_2.1.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![pyg](https://img.shields.io/badge/PyG_2.4+-3C2179?logo=pyg&logoColor=#3C2179)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

**Starter code**: Structural awareness tests for graph transformers

## Install
I recommend you use a `conda` environment (or similar). E.g., run

```bash
conda create -n structural-awareness-gt python=3.10
conda activate structural-awareness-gt
```

then install the only two dependencies
> NOTE: Tested on Linux. Look up specific installation instructions for your setup
> for [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) 

```bash
pip install torch torch_geometric
```

## Run
Simply run

```bash
python main.py
```
