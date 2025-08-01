from typing import Iterable, Tuple

import networkx as nx
import numpy as np

__all__ = ["add_ground_truth_to_edge_attrs", "get_directed_edge_indices"]


def add_ground_truth_to_edge_attrs(graph: nx.Graph, gt: np.ndarray) -> nx.Graph:
    var_dict = {}
    for edge, val in zip(graph.edges, gt):
        var_dict[edge] = val.item()

    nx.set_edge_attributes(graph, var_dict, "gt")

    return graph


def get_directed_edge_indices(
    graph: nx.Graph, cycles: Iterable[Iterable[Tuple[int, int]]]
) -> list[list[int]]:
    di_graph = graph.to_directed()
    di_graph_edges = list(di_graph.edges)

    directed_cycles = []

    for cycle in cycles:
        ind_cycle = []
        for u, v in cycle:
            ind_cycle.append(di_graph_edges.index((u, v)))
        directed_cycles.append(ind_cycle)

    return directed_cycles
