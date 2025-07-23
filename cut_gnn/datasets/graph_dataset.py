import glob
import os
import pickle
import re
from typing import Iterable, Optional, Sequence

import networkx as nx
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_add

from cut_gnn.utils import path_type, read_graph

__all__ = ["GraphDataset", "GraphDataModule"]


# Source: https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
def sorted_alphanumeric(data):
    def convert(text: str):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


class GraphDataset(Dataset):
    def __init__(
        self,
        root_path: path_type,
        load_cycles: bool = True,
        max_cycles_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        all_pickle_files = glob.glob(os.path.join(root_path, "*.pkl"))

        cycles_files = []
        graph_files = []
        for file in all_pickle_files:
            if file.endswith("cycles.pkl") and load_cycles:
                cycles_files.append(file)
            else:
                graph_files.append(file)

        self.__graph_files = sorted_alphanumeric(graph_files)
        self.__cycles_files = sorted_alphanumeric(cycles_files)
        self.load_cycles = load_cycles
        self.max_cycles_len = max_cycles_len

    def __len__(self):
        return len(self.__graph_files)

    def __getitem__(self, idx: int) -> Data:
        G = read_graph(self.__graph_files[idx])

        G = G.to_directed() if not nx.is_directed(G) else G
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        has_gt = "gt" in next(iter(G.edges(data=True)))[-1]

        mapping = dict(zip(G.nodes(), range(num_nodes)))
        edge_index = torch.empty((2, num_edges), dtype=torch.long)
        weights = torch.empty(num_edges, dtype=torch.float32)
        if has_gt:
            gt = torch.empty(num_edges, dtype=torch.float32)
        for i, (src, dst, edge_attrs) in enumerate(G.edges(data=True)):
            edge_index[0, i] = mapping[src]
            edge_index[1, i] = mapping[dst]
            weights[i] = edge_attrs["weight"]
            if has_gt:
                gt[i] = edge_attrs["gt"]

        positive_edge_mask = weights > 0
        negative_edge_mask = weights < 0

        pos_weights = weights[positive_edge_mask]
        neg_weights = weights[negative_edge_mask]

        pos_edge_index = edge_index[:, positive_edge_mask][0]
        neg_edge_index = edge_index[:, negative_edge_mask][0]

        pos_w_sum = scatter_add(pos_weights, pos_edge_index, dim=0, dim_size=num_nodes)
        neg_w_sum = scatter_add(neg_weights, neg_edge_index, dim=0, dim_size=num_nodes)
        x = torch.stack([pos_w_sum, neg_w_sum], dim=1)

        data_dict = {"edge_index": edge_index, "edge_weight": weights, "x": x}
        if has_gt:
            data_dict["y"] = gt

        if self.load_cycles:
            with open(self.__cycles_files[idx], "rb") as file:
                cycles = pickle.load(file)
                if self.max_cycles_len:
                    cycles = list(
                        filter(lambda c: len(c) <= self.max_cycles_len, cycles)
                    )
                data_dict["cycles"] = cycles

        data = Data.from_dict(data_dict)

        return data


def cycles_collate_fn(data_list: Iterable[Data]) -> Batch:
    batch = Batch.from_data_list(data_list, exclude_keys=["cycles"])

    batched_cycles = data_list[0].cycles
    offset = data_list[0].edge_index.size(1)

    for data in data_list[1:]:
        for cycle in data.cycles:
            shifted_indices = [idx + offset for idx in cycle]
            batched_cycles.append(shifted_indices)
        offset += data.edge_index.size(1)

    batch.cycles = batched_cycles
    return batch


class GraphDataModule(LightningDataModule):
    def __init__(
        self,
        root_path: path_type,
        batch_size: int = 32,
        fractions: Sequence[float | int] = (0.8, 0.1, 0.1),
        num_workers: int = 16,
        load_cycles: bool = True,
        max_cycles_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert len(fractions) == 3, (
            "You have to specify 3 fractions for train, val and test"
        )

        self.root_path = root_path
        self.batch_size = batch_size
        self.load_cycles = load_cycles
        self.max_cycles_len = max_cycles_len
        self.fractions = fractions
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        ds = GraphDataset(self.root_path, self.load_cycles, self.max_cycles_len)
        rnd_gen = torch.Generator().manual_seed(42)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            ds, self.fractions, generator=rnd_gen
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=cycles_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=cycles_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=cycles_collate_fn,
        )
