import numpy as np
import networkx as nx
import cvxpy as cp

import typing
from itertools import pairwise, combinations

def to_ilp(graph: nx.Graph, cycle_lenght_bound: int = None, ret_cycles: bool = True) -> cp.Problem:
    var = cp.Variable(graph.number_of_edges(), boolean=True)
    c = list(nx.get_edge_attributes(graph, 'weight').values())
    obj = cp.Minimize(sum(cp.multiply(c, var)))

    edges = list(graph.edges)

    constraints = []
    cycles = []
    for cycle in nx.simple_cycles(graph, cycle_lenght_bound):
        indexes = {} # map from node idxs in a cycle to idx of edge in var
        cycle_ext = cycle.copy()
        cycle_ext.append(cycle[0])

        if ret_cycles:
            edge_cycle = list(pairwise(cycle_ext))
            cycles.append(edge_cycle)
        else:
            edge_cycle = pairwise(cycle_ext)

        for idx, (u, v) in zip(cycle, edge_cycle):
            uv = (u, v) if u < v else (v, u)
            indexes[idx] = edges.index(uv)

        rcycle = reversed(cycle)
        for e, rest in zip(cycle, combinations(rcycle, len(cycle)-1)):
            constr_sum = sum([1 - var[indexes[er]] for er in rest])
            constraints.append(1 - var[indexes[e]] <= constr_sum)

    problem = cp.Problem(obj, constraints=constraints)

    return problem, cycles

def solve_ilp(problem: cp.Problem, solver: typing.Optional[str] = None) -> np.array:
    problem.solve(solver=solver)
    vars = problem.variables()[0].value
    return vars