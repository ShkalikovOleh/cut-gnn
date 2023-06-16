from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import numpy as np
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, random_split
from lightning import LightningDataModule

import os
import glob
import pickle
import re
from typing import Sequence

from utils import read_graph, path_type

__all__ = ['GraphDataset', 'graph_batch_to_device', 'GraphDataModule']

# Source: https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

class GraphDataset(Dataset):
    def __init__(self, root_path: path_type, load_cycles: bool = True) -> None:
        super().__init__()

        self.__graph_files = sorted_alphanumeric(glob.glob(os.path.join(root_path, '*.gml')))
        self.load_cycles = load_cycles
        if load_cycles:
            self.__cycles_files = sorted_alphanumeric(glob.glob(os.path.join(root_path, '*.pkl')))

    def __len__(self):
        return len(self.__graph_files)

    def __getitem__(self, idx: int) -> Data:
        G = read_graph(self.__graph_files[idx])
        data = torch_geometric.utils.from_networkx(G)
        data.x = data.x.float()
        data.num_nodes = G.number_of_nodes()

        if self.load_cycles:
            with open(self.__cycles_files[idx], 'rb') as file:
                cycles = pickle.load(file)
                for k in cycles.keys():
                    cycles[k] = torch.tensor(np.array(cycles[k]))
                data.cycles = [cycles]

        return data


def graph_batch_to_device(batch: Batch, device: torch.device, has_cycles: bool = True) -> Batch:
    batch.x = batch.x.to(device)
    batch.edge_index = batch.edge_index.to(device)
    batch.weight = batch.weight.to(device)
    batch.gt = batch.gt.to(device)
    # batch.batch = batch.batch.to(device)
    # batch.ptr = batch.ptr.to(device)

    if has_cycles:
        device_cycles = []
        for g_cycle in batch.cycles:
            g_dev_cycles = {}
            for l, l_cycles in g_cycle[0].items():
                g_dev_cycles[l] = l_cycles.to(device)
            device_cycles.append([g_dev_cycles])
        batch.cycles = device_cycles

    return batch


class GraphDataModule(LightningDataModule):
    def __init__(self, root_path: path_type, batch_size: int,
                  fractions: Sequence[float|int], num_workers: int = 2,
                  load_cycles: bool = True) -> None:
        super().__init__()
        assert len(fractions) == 3, 'You have specify 3 fractions for train, val and test'

        self.root_path = root_path
        self.batch_size = batch_size
        self.load_cycles = load_cycles
        self.fractions = fractions
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        ds = GraphDataset(self.root_path, self.load_cycles)
        self.train_ds, self.val_ds, self.test_ds = random_split(ds, self.fractions)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, self.batch_size, num_workers=self.num_workers)

    def transfer_batch_to_device(self, batch: Batch, device:
                                 torch.device, dataloader_idx: int) -> Batch:
        return graph_batch_to_device(batch, device, self.load_cycles)
