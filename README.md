# cut-gnn
This repository is a seminar project developing the ideas from the paper:

```
Jung, S. and Keuper, M., 2022. Learning to solve minimum cost multicuts efficiently using edge-weighted graph convolutional neural networks.
```

Please note that this project is **not properly tested and not currently maintained**.
This project was developed out of my own interest and initiative, going beyond the strict requirements of the seminar, and should be treated as such.

## New Ideas

* Unsupervised formulation
* Relaxed cycle consistency loss
* Weight embeddings
* [**NOT Implemented**] Learning orthogonal embedding

For detailed explanations of the paper and my ideas, please refer to the `report` branch.

## Installation

To get started, first clone the repository to your local machine:
```bash
git clone git@github.com:ShkalikovOleh/cut-gnn.git
```

This project utilizes `uv` for dependency management. You will need to install `uv` first if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install the project dependencies, navigate to the root of this repository and run one of the following commands based on your CUDA setup:
* **For CUDA 11.8:**
  ```bash
  uv sync --extra "cu118"
  ```
* **For CPU-only development:**
  ```bash
  uv sync --extra "cpu"
  ```

## Usage

Scripts for generating IrisMP dataset and running training/testing are available in the [`scripts`](scripts) directory.

This project heavily relies on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). It is highly recommended to read the PyTorch Lightning documentation carefully, especially regarding `LightningCLI`, as the scripts in this repository make extensive use of it.