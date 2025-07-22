## Please read the docs: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html

from lightning.pytorch.cli import LightningCLI

from cut_gnn.datasets import GraphDataModule
from cut_gnn.supervised import SupervisedMultiCut


def main():
    cli = LightningCLI(model_class=SupervisedMultiCut, datamodule_class=GraphDataModule)  # noqa: F841


if __name__ == "__main__":
    main()
