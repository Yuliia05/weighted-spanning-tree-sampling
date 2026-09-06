from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Edge:
    id: int
    u: int
    v: int
    weight: float


class GraphState:
    def __init__(
        self,
        n: int,
        edges: Sequence[Edge],
    ):
        self.n_original = n
        self.edges = list(edges)

        self._adjacency: List[List[Tuple[int, Edge]]] = [
            [] for _ in range(n)
        ]
        self._weighted_degrees = np.zeros(n, dtype=float)

        for edge in self.edges:
            self._adjacency[edge.u].append((edge.v, edge))
            self._adjacency[edge.v].append((edge.u, edge))
            self._weighted_degrees[edge.u] += edge.weight
            self._weighted_degrees[edge.v] += edge.weight

        self._cumulative_weights: List[np.ndarray] = []

        for v in range(n):
            cumulative = np.cumsum(
                [edge.weight for _, edge in self._adjacency[v]],
                dtype=float,
            )
            self._cumulative_weights.append(cumulative)

    def neighbors(self, v: int) -> List[Tuple[int, Edge]]:
        return self._adjacency[v]

    def weighted_degree(self, v: int) -> float:
        return float(self._weighted_degrees[v])

    def cumulative_weights(self, v: int) -> np.ndarray:
        return self._cumulative_weights[v]


def make_graph(
    n: int,
    weighted_edges: Iterable[Tuple[int, int, float]],
) -> GraphState:

    edges: List[Edge] = []
    seen = set()

    for edge_id, (u, v, weight) in enumerate(weighted_edges):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("Edge endpoint outside 0,...,n-1.")

        if u == v:
            raise ValueError("Input graph must not contain self-loops.")

        if weight <= 0:
            raise ValueError("All edge weights must be positive.")

        key = tuple(sorted((u, v)))

        if key in seen:
            raise ValueError("Input graph must be simple.")

        seen.add(key)
        edges.append(Edge(edge_id, u, v, float(weight)))

    state = GraphState(n, edges)

    if n > 1 and not is_connected(state):
        raise ValueError("Input graph must be connected.")

    return state


def is_connected(state: GraphState) -> bool:
    n = state.n_original

    if n <= 1:
        return True

    visited = [False] * n
    visited[0] = True
    stack = [0]

    while stack:
        v = stack.pop()

        for neighbor, edge in state.neighbors(v):
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)

    return all(visited)


def sample_neighbor(
    state: GraphState,
    vertex: int,
    rng: np.random.Generator,
) -> Tuple[int, Edge]:

    neighbors = state.neighbors(vertex)

    if not neighbors:
        raise RuntimeError("Current vertex has no neighbors.")

    cumulative = state.cumulative_weights(vertex)
    total_weight = state.weighted_degree(vertex)

    r = rng.random() * total_weight
    index = int(np.searchsorted(cumulative, r, side="right"))

    # Safety against a possible floating-point boundary case.
    if index >= len(neighbors):
        index = len(neighbors) - 1

    return neighbors[index]


def sample_weighted_spanning_tree(
    state: GraphState,
    start_vertex: int = 0,
    rng: Optional[np.random.Generator] = None,
    return_steps: bool = False,
) -> Union[List[Edge], Tuple[List[Edge], int]]:

    if rng is None:
        rng = np.random.default_rng()

    n = state.n_original

    if not (0 <= start_vertex < n):
        raise ValueError("Start vertex outside 0,...,n-1.")

    if n <= 1:
        if return_steps:
            return [], 0
        return []

    visited = [False] * n
    visited[start_vertex] = True

    X = start_vertex
    chosen: List[Edge] = []
    s = 1
    steps = 0

    while s < n:
        Y, edge = sample_neighbor(state, X, rng)
        steps += 1

        if not visited[Y]:
            chosen.append(edge)
            visited[Y] = True
            s += 1

        X = Y

    if len(chosen) != n - 1:
        raise RuntimeError(
            f"Sampler returned {len(chosen)} edges; expected {n - 1}."
        )

    if return_steps:
        return chosen, steps

    return chosen
