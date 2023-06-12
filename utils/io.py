import numpy as np
import networkx as nx
from networkx.readwrite.gml import literal_destringizer, literal_stringizer

import pathlib
import typing

__all__ = ['write_graph', 'read_graph', 'path_type']

path_type = typing.Union[str, pathlib.Path]

def write_graph(graph: nx.Graph, path: path_type) -> None:
    nx.write_gml(graph, path, stringizer=np.array2string)

def read_graph(path: path_type) -> nx.Graph:
    def str_to_val(string):
        if '[' in string:
            string = string.replace('[','')
            string = string.replace(']','')
            res = np.fromstring(string, sep=' ')
            return res
        else:
            return literal_destringizer(string)

    return nx.read_gml(path, destringizer=str_to_val)