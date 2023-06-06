import torch_geometric
from torch.utils.data import Dataset

import os
import glob
import pickle

from utils.io import read_graph, path_type

class GraphDataset(Dataset):
    def __init__(self, root_path: path_type, load_cycles: bool = True) -> None:
        super().__init__()

        self.__graph_files = sorted(glob.glob(os.path.join(root_path, '*.gml')))
        self.load_cycles = load_cycles
        if load_cycles:
            self.__cycles_files = sorted(glob.glob(os.path.join(root_path, '*.pkl')))

    def __len__(self):
        return len(self.__graph_files)

    def __getitem__(self, idx):
        G = read_graph(self.__graph_files[idx])
        data = torch_geometric.utils.from_networkx(G)
        data.x = data.x.float()
        data.num_nodes = G.number_of_nodes()

        if self.load_cycles:
            with open(self.__cycles_files[idx], 'rb') as file:
                cycles = pickle.load(file)
                data.cycles = cycles

        return data
