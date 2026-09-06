from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import math

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
        vertices: Optional[np.ndarray] = None,
        active: Optional[np.ndarray] = None,
    ):
        self.n_original = n
        self.edges = list(edges)
        self.vertices = (
            np.arange(n, dtype=int) if vertices is None else vertices.copy()
        )
        self.active = (
            np.ones(len(edges), dtype=bool) if active is None else active.copy()
        )

    def copy(self) -> "GraphState":
        return GraphState(
            self.n_original,
            self.edges,
            vertices=self.vertices,
            active=self.active,
        )

    def supernode(self, v: int) -> int:
        return int(self.vertices[v])

    def current_nodes(self) -> List[int]:
        return sorted(set(map(int, self.vertices)))

    def num_nodes(self) -> int:
        return len(set(map(int, self.vertices)))

    def is_loop(self, edge: Edge) -> bool:
        return self.supernode(edge.u) == self.supernode(edge.v)

    def next_edge(self) -> Optional[Edge]:
        """First active non-loop edge in the fixed original ordering."""
        for edge in self.edges:
            if self.active[edge.id] and not self.is_loop(edge):
                return edge
        return None

    def delete(self, edge: Edge) -> None:
        self.active[edge.id] = False

    def contract(self, edge: Edge) -> None:
        """Contract edge and remove it from future processing."""
        a = self.supernode(edge.u)
        b = self.supernode(edge.v)
        if a == b:
            self.active[edge.id] = False
            return

        keep, remove = min(a, b), max(a, b)
        self.vertices[self.vertices == remove] = keep
        self.active[edge.id] = False

    def laplacian(self) -> np.ndarray:
        nodes = self.current_nodes()
        index = {node: i for i, node in enumerate(nodes)}
        L = np.zeros((len(nodes), len(nodes)), dtype=float)

        for edge in self.edges:
            if not self.active[edge.id]:
                continue

            a = self.supernode(edge.u)
            b = self.supernode(edge.v)

            if a == b:
                continue

            i, j = index[a], index[b]
            w = edge.weight

            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w

        return L


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
            raise ValueError(
                "Input graph must be simple. Parallel edges may appear only "
                "after contractions."
            )
        seen.add(key)

        edges.append(Edge(edge_id, u, v, float(weight)))

    state = GraphState(n, edges)
    if n > 1 and tau_weighted(state) <= 0:
        raise ValueError("Input graph must be connected.")
    return state


def tau_weighted(state: GraphState) -> float:
    L = state.laplacian()
    n = L.shape[0]

    if n <= 1:
        return 1.0

    reduced = L[:-1, :-1]
    det = float(np.linalg.det(reduced))

    scale = max(1.0, float(np.max(np.abs(reduced))) ** max(1, n - 1))
    tol = 1e-12 * scale
    if det < 0 and abs(det) <= tol:
        det = 0.0

    return det


def deletion_contraction_probability(
    state: GraphState,
    edge: Edge,
) -> Tuple[float, float, float]:
    
    if state.is_loop(edge):
        return 0.0, tau_weighted(state), 0.0

    deleted = state.copy()
    deleted.delete(edge)
    tau_delete = tau_weighted(deleted)

    contracted = state.copy()
    contracted.contract(edge)
    tau_contract = tau_weighted(contracted)

    numerator = edge.weight * tau_contract
    denominator = tau_delete + numerator

    if denominator <= 0:
        raise RuntimeError(
            "Deletion/contraction identity produced zero total tree mass."
        )

    p = numerator / denominator

    p = min(1.0, max(0.0, p))
    return p, tau_delete, tau_contract


def sample_weighted_spanning_tree(
    state: GraphState,
    rng: Optional[np.random.Generator] = None,
) -> List[Edge]:
    
    if rng is None:
        rng = np.random.default_rng()

    current = state.copy()
    chosen: List[Edge] = []

    while current.num_nodes() > 1:
        edge = current.next_edge()
        if edge is None:
            raise RuntimeError(
                "No non-loop edge remains before contraction reached one vertex."
            )

        p, _, _ = deletion_contraction_probability(current, edge)

        if rng.random() <= p:
            chosen.append(edge)
            current.contract(edge)
        else:
            current.delete(edge)

    if len(chosen) != state.n_original - 1:
        raise RuntimeError(
            f"Sampler returned {len(chosen)} edges; expected "
            f"{state.n_original - 1}."
        )

    return chosen


