import concurrent.futures
import os
import pickle
import time
import typing
from functools import partial
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.special import logit

from cut_gnn.utils import (
    add_ground_truth_to_edge_attrs,
    get_directed_edge_indices,
    path_type,
    solve_ilp,
    to_ilp,
    write_graph,
)

__all__ = [
    "load_iris_df",
    "generate_iris_graph",
    "generate_iris_sample",
    "generate_iris_dataset",
]


def load_iris_df(path: path_type) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.drop("variety", axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def generate_iris_graph(
    iris_df: pd.DataFrame,
    n_nodes_min: int,
    n_nodes_max: int,
    sigma: float = 1.0,
    n_sample_feat: int = 2,
) -> nx.Graph:
    column_idxs = [0, 1, 2, 3]
    col_idxs = np.random.choice(column_idxs, n_sample_feat)
    n_nodes = np.random.randint(n_nodes_min, n_nodes_max)
    row_idxs = np.random.choice(range(len(iris_df)), n_nodes, replace=False)

    distances = pdist(iris_df.iloc[row_idxs, col_idxs], metric="euclidean")
    similarities = np.exp(-np.power(distances, 2) / (2 * sigma * sigma))
    weights = squareform(logit(np.clip(similarities, 10**-6, 1 - 10**-6)))

    graph = nx.Graph()
    for i, j in combinations(range(n_nodes), 2):
        graph.add_edge(i, j, weight=weights[i, j].item())

    return graph


def generate_iris_sample(
    sample_idx: int,
    out_path: path_type,
    iris_df: pd.DataFrame,
    n_nodes_min: int,
    n_nodes_max: int,
    sigma: float = 1.0,
    n_sample_feat: int = 2,
    include_cycles: bool = True,
    solver: typing.Optional[str] = None,
) -> None:
    np.random.seed(
        (os.getpid() * int(time.time())) % 123456789
    )  # different seed for different threads

    G = generate_iris_graph(iris_df, n_nodes_min, n_nodes_max, sigma, n_sample_feat)

    problem, cycles = to_ilp(G, 3, ret_cycles=include_cycles)
    gt_edge = solve_ilp(problem, solver=solver)

    add_ground_truth_to_edge_attrs(G, gt_edge)
    write_graph(G, os.path.join(out_path, f"iris_{sample_idx}.pkl"))

    if include_cycles:
        ind_edge_idxs = get_directed_edge_indices(G, cycles)
        with open(
            os.path.join(out_path, f"iris_{sample_idx}_cycles.pkl"), "wb"
        ) as file:
            pickle.dump(ind_edge_idxs, file)  # save dict: length -> list of edge idxs


def generate_iris_dataset(
    n_graphs: int,
    out_path: path_type,
    iris_df: pd.DataFrame,
    n_nodes_min: int,
    n_nodes_max: int,
    sigma: float = 0.6,
    n_sample_feat: int = 2,
    include_cycles: bool = True,
    max_workers: typing.Optional[int] = None,
    solver: typing.Optional[str] = None,
    start_idx: int = 0,
    chunksize: int = 16,
) -> None:
    generator = partial(
        generate_iris_sample,
        out_path=out_path,
        iris_df=iris_df,
        n_nodes_min=n_nodes_min,
        n_nodes_max=n_nodes_max,
        sigma=sigma,
        n_sample_feat=n_sample_feat,
        include_cycles=include_cycles,
        solver=solver,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            executor.map(
                generator, range(start_idx, start_idx + n_graphs), chunksize=chunksize
            )
        )
