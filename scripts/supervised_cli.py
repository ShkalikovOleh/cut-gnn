# Please read the docs: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
# Example usage: python scripts/supervised_cli.py fit/test [all_your_params] OR --config [yaml file]

from lightning.pytorch.cli import LightningCLI

from cut_gnn.datasets import GraphDataModule
from cut_gnn.supervised import SupervisedMultiCut


def main():
    cli = LightningCLI(model_class=SupervisedMultiCut, datamodule_class=GraphDataModule)  # noqa: F841


if __name__ == "__main__":
    main()
