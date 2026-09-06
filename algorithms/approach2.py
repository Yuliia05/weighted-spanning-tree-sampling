from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
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
        return len(self.current_nodes())

    def is_loop(self, edge: Edge) -> bool:
        return self.supernode(edge.u) == self.supernode(edge.v)

    def next_edge(self) -> Optional[Edge]:
        for edge in self.edges:
            if self.active[edge.id] and not self.is_loop(edge):
                return edge
        return None

    def delete(self, edge: Edge) -> None:
        self.active[edge.id] = False

    def reduced_coordinates(self, root: int) -> dict:
        nodes = [v for v in self.current_nodes() if v != root]
        return {v: i for i, v in enumerate(nodes)}

    def contract(self, edge: Edge, root: int) -> int:
        u = self.supernode(edge.u)
        v = self.supernode(edge.v)

        if u == v:
            self.active[edge.id] = False
            return root

        if u == root:
            keep, remove = root, v
        elif v == root:
            keep, remove = root, u
        else:
            coordinates = self.reduced_coordinates(root)
            i = coordinates[u]
            j = coordinates[v]

            if i <= j:
                keep, remove = u, v
            else:
                keep, remove = v, u

        self.vertices[self.vertices == remove] = keep
        self.active[edge.id] = False

        for current_edge in self.edges:
            if self.active[current_edge.id] and self.is_loop(current_edge):
                self.active[current_edge.id] = False

        return root

    def reduced_laplacian(self, root: int) -> np.ndarray:
        nodes = self.current_nodes()
        index = {node: i for i, node in enumerate(nodes)}
        L = np.zeros((len(nodes), len(nodes)), dtype=float)

        for edge in self.edges:
            if not self.active[edge.id]:
                continue

            u = self.supernode(edge.u)
            v = self.supernode(edge.v)

            if u == v:
                continue

            i = index[u]
            j = index[v]
            w = edge.weight

            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w

        root_index = index[root]
        return np.delete(np.delete(L, root_index, axis=0), root_index, axis=1)

    def incidence_vector(self, edge: Edge, root: int) -> np.ndarray:
        coordinates = self.reduced_coordinates(root)
        b = np.zeros(self.num_nodes() - 1, dtype=float)

        u = self.supernode(edge.u)
        v = self.supernode(edge.v)

        if u != root:
            b[coordinates[u]] += 1.0

        if v != root:
            b[coordinates[v]] -= 1.0

        return b


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

    adjacency = [[] for _ in range(n)]

    for edge in state.edges:
        adjacency[edge.u].append(edge.v)
        adjacency[edge.v].append(edge.u)

    visited = [False] * n
    visited[0] = True
    stack = [0]

    while stack:
        v = stack.pop()

        for u in adjacency[v]:
            if not visited[u]:
                visited[u] = True
                stack.append(u)

    return all(visited)


def cholesky_factorization(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    U = np.zeros_like(A, dtype=float)

    for k in range(n):
        value = A[k, k]

        for j in range(k):
            value -= U[j, k] ** 2

        if value <= 0:
            if value > -1e-12:
                value = 0.0
            else:
                raise RuntimeError("Cholesky pivot is not positive.")

        U[k, k] = np.sqrt(value)

        if U[k, k] == 0:
            raise RuntimeError("Cholesky factorization produced a zero pivot.")

        for ell in range(k + 1, n):
            value = A[k, ell]

            for j in range(k):
                value -= U[j, k] * U[j, ell]

            U[k, ell] = value / U[k, k]

    return U


def forward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    y = np.zeros(n, dtype=float)

    for i in range(n):
        value = b[i]

        for j in range(i):
            value -= U[j, i] * y[j]

        y[i] = value / U[i, i]

    return y


def edge_inclusion_probability(
    U: np.ndarray,
    b: np.ndarray,
    weight: float,
) -> float:

    y = forward_substitution(U, b)
    p = float(weight * np.dot(y, y))

    if p < 0:
        if p > -1e-12:
            p = 0.0
        else:
            raise RuntimeError("Computed a negative edge probability.")

    if p > 1:
        if p < 1.0 + 1e-10:
            p = 1.0
        else:
            raise RuntimeError("Computed an edge probability larger than one.")

    return p


def chol_downdate(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    U = U.copy()
    x = x.copy()
    d = len(x)

    for k in range(d):
        value = U[k, k] ** 2 - x[k] ** 2

        if value <= 0:
            if value > -1e-12:
                value = 0.0
            else:
                raise RuntimeError("Cholesky downdate pivot is not positive.")

        rho = np.sqrt(value)

        if rho == 0:
            raise RuntimeError("Cholesky downdate produced a zero pivot.")

        c = rho / U[k, k]
        s = x[k] / U[k, k]

        U[k, k] = rho
        x[k] = 0.0

        for j in range(k + 1, d):
            U[k, j] = (U[k, j] - s * x[j]) / c
            x[j] = c * x[j] - s * U[k, j]

    return U


def givens_zero(
    B: np.ndarray,
    p: int,
    q: int,
    ell: int,
) -> np.ndarray:

    if B[q, ell] != 0:
        rho = np.hypot(B[p, ell], B[q, ell])

        c = B[p, ell] / rho
        s = B[q, ell] / rho

        for t in range(B.shape[1]):
            z = c * B[p, t] + s * B[q, t]
            B[q, t] = -s * B[p, t] + c * B[q, t]
            B[p, t] = z

    return B


def contract_factor(
    U: np.ndarray,
    state: GraphState,
    u: int,
    v: int,
    root: int,
) -> np.ndarray:

    d = U.shape[0]
    coordinates = state.reduced_coordinates(root)

    if u == root or v == root:
        other = v if u == root else u
        j = coordinates[other]

        B = np.delete(U, j, axis=1)

        for k in range(j, d - 1):
            B = givens_zero(B, k, k + 1, k)

    else:
        i = coordinates[u]
        j = coordinates[v]

        if i > j:
            i, j = j, i

        B = U.copy()
        B[:, i] = B[:, i] + B[:, j]
        B = np.delete(B, j, axis=1)

        for k in range(j, i, -1):
            B = givens_zero(B, k - 1, k, i)

        for k in range(i + 1, d - 1):
            B = givens_zero(B, k, k + 1, k)

    R = B[: d - 1, :].copy()

    for k in range(d - 1):
        if R[k, k] < 0:
            R[k, :] *= -1.0

        if abs(R[k, k]) < 1e-14:
            raise RuntimeError("ContractFactor produced a zero diagonal entry.")

    return R


def sample_weighted_spanning_tree(
    state: GraphState,
    root: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> List[Edge]:

    if rng is None:
        rng = np.random.default_rng()

    if not (0 <= root < state.n_original):
        raise ValueError("Root vertex outside 0,...,n-1.")

    if state.n_original <= 1:
        return []

    current = state.copy()
    chosen: List[Edge] = []

    A = current.reduced_laplacian(root)
    U = cholesky_factorization(A)

    while current.num_nodes() > 1:
        edge = current.next_edge()

        if edge is None:
            raise RuntimeError(
                "No non-loop edge remains before contraction reached one vertex."
            )

        u = current.supernode(edge.u)
        v = current.supernode(edge.v)

        b = current.incidence_vector(edge, root)
        p = edge_inclusion_probability(U, b, edge.weight)

        xi = rng.random()

        if xi <= p:
            chosen.append(edge)

            U = contract_factor(U, current, u, v, root)
            root = current.contract(edge, root)

        else:
            x = np.sqrt(edge.weight) * b
            U = chol_downdate(U, x)
            current.delete(edge)

    if len(chosen) != state.n_original - 1:
        raise RuntimeError(
            f"Sampler returned {len(chosen)} edges; expected "
            f"{state.n_original - 1}."
        )

    return chosen
