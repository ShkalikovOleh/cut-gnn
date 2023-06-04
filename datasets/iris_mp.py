import numpy as np
import pandas as pd
import networkx as nx
import torch_geometric
from scipy.spatial.distance import pdist, squareform
from scipy.special import logit
from torch.utils.data import Dataset

from itertools import combinations
from functools import partial
import concurrent.futures
import os
import typing
import pickle
import time
import glob

from utils.io import write_graph, read_graph, path_type
from utils.ilp import to_ilp, solve_ilp
from utils.nx import add_ground_truth_to_edge_attrs, get_undirected_edge_idx

def generate_iris_graph(iris_df : pd.DataFrame, n_nodes_low : int, n_nodes_high : int,
                        sigma : float = 1.0, n_sample_feat : int = 2, gen_x: bool = True) -> nx.Graph:
    column_idxs = [0, 1, 2, 3]
    col_idxs = np.random.choice(column_idxs, n_sample_feat)
    n_nodes = np.random.randint(n_nodes_low, n_nodes_high)
    row_idxs = np.random.choice(range(len(iris_df)), n_nodes, replace=False)

    distances = pdist(iris_df.iloc[row_idxs, col_idxs], metric='euclidean')
    similarities = np.exp(-np.power(distances, 2) / (2*sigma*sigma))
    weights = squareform(logit(np.clip(similarities, 10**-6, 1-10**-6)))

    graph = nx.Graph()
    for i, j in combinations(range(n_nodes), 2):
        graph.add_edge(i, j, weight=weights[i, j])

    if gen_x:
        pos_sum = np.sum(np.where(weights > 0, weights, 0), axis=1)
        neg_sum = np.sum(np.where(weights <= 0, weights, 0), axis=1)
        for node in graph.nodes:
            graph.add_node(node, x = np.asarray([pos_sum[node], neg_sum[node]]))

    return graph

def generate_iris_sample(i_sample: int, path: path_type, iris_df: pd.DataFrame, n_nodes_low : int,
                          n_nodes_high : int, sigma : float = 1.0, n_sample_feat : int = 2,
                          gen_x: bool = True, include_cycles: bool = True,
                          solver:typing.Optional[str] = None) -> None:
    np.random.seed((os.getpid() * int(time.time())) % 123456789) # different seed for different threads

    G = generate_iris_graph(iris_df, n_nodes_low, n_nodes_high, sigma, n_sample_feat, gen_x)

    problem, cycles = to_ilp(G, 3, ret_cycles=include_cycles)
    gt_edge = solve_ilp(problem, solver=solver)

    add_ground_truth_to_edge_attrs(G, gt_edge)
    write_graph(G, os.path.join(path, f'iris_{i_sample}.gml'))

    if include_cycles:
        ind_edge_idxs = get_undirected_edge_idx(G, cycles)
        with open(os.path.join(path, f'iris_{i_sample}_cycles.pkl'), 'wb') as file:
            pickle.dump(ind_edge_idxs, file)

def generate_iris_dataset(n_graphs: int, path: path_type, iris_df: pd.DataFrame, n_nodes_low : int,
                          n_nodes_high : int, sigma : float = 1.0, n_sample_feat : int = 2,
                          gen_x: bool = True, include_cycles: bool = True,
                          max_workers: typing.Optional[int] = None,
                          solver:typing.Optional[str] = None) -> None:
    generator = partial(generate_iris_sample, path = path, iris_df = iris_df, n_nodes_low = n_nodes_low,
                          n_nodes_high = n_nodes_high, sigma = sigma, n_sample_feat = n_sample_feat,
                          gen_x = gen_x, include_cycles=include_cycles, solver=solver)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(generator, range(n_graphs))

class IrisMPDataset(Dataset):
    def __init__(self, root_path: path_type, load_cycles: bool = True) -> None:
        super().__init__()

        self.__graph_files = glob.glob(os.path.join(root_path, '*.gml'))
        self.load_cycles = load_cycles
        if load_cycles:
            self.__cycles_files = glob.glob(os.path.join(root_path, '*.pkl'))

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

