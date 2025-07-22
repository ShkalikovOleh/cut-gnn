import pathlib
import pickle
import typing

import networkx as nx

__all__ = ["write_graph", "read_graph", "path_type"]

path_type = typing.Union[str, pathlib.Path]


def write_graph(graph: nx.Graph, path: path_type) -> None:
    with open(path, "wb") as file:
        pickle.dump(graph, file)


def read_graph(path: path_type) -> nx.Graph:
    with open(path, "rb") as file:
        G = pickle.load(file)
    return G
