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

    def component(self, v: int) -> int:
        return int(self.vertices[v])

    def is_loop(self, edge: Edge) -> bool:
        return self.component(edge.u) == self.component(edge.v)

    def delete(self, edge: Edge) -> None:
        self.active[edge.id] = False

    def contract(self, edge: Edge, root: int) -> None:
        u = self.component(edge.u)
        v = self.component(edge.v)

        if u == v:
            self.active[edge.id] = False
            return

        root_component = self.component(root)

        if u == root_component:
            keep, remove = u, v
        elif v == root_component:
            keep, remove = v, u
        elif u < v:
            keep, remove = u, v
        else:
            keep, remove = v, u

        self.vertices[self.vertices == remove] = keep
        self.active[edge.id] = False

        for current_edge in self.edges:
            if self.active[current_edge.id] and self.is_loop(current_edge):
                self.active[current_edge.id] = False

    def edge_between(self, u: int, v: int) -> Optional[Edge]:
        for edge in self.edges:
            if not self.active[edge.id]:
                continue

            if (edge.u == u and edge.v == v) or (edge.u == v and edge.v == u):
                return edge

        return None

    def reduced_laplacian(self, root: int) -> np.ndarray:
        L = np.zeros((self.n_original, self.n_original), dtype=float)

        for edge in self.edges:
            u = edge.u
            v = edge.v
            w = edge.weight

            L[u, u] += w
            L[v, v] += w
            L[u, v] -= w
            L[v, u] -= w

        return np.delete(np.delete(L, root, axis=0), root, axis=1)

    def reduced_incidence_vector(self, edge: Edge, root: int) -> np.ndarray:
        b = np.zeros(self.n_original - 1, dtype=float)

        if edge.u != root:
            b[reduced_coordinate(edge.u, root)] += 1.0

        if edge.v != root:
            b[reduced_coordinate(edge.v, root)] -= 1.0

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


def reduced_coordinate(v: int, root: int) -> int:
    if v < root:
        return v
    return v - 1


def block_indices(vertices: Sequence[int], root: int) -> List[int]:
    return [
        reduced_coordinate(v, root)
        for v in vertices
        if v != root
    ]


def recursive_spd_inverse(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=float)

    if n == 1:
        value = float(A[0, 0])

        if value <= 0:
            if value > -1e-12:
                value = 1e-12
            else:
                raise RuntimeError("SPD inversion received a non-positive pivot.")

        return np.array([[1.0 / value]], dtype=float)

    k = n // 2

    B = A[:k, :k]
    C = A[:k, k:]
    E = A[k:, k:]

    B_inv = recursive_spd_inverse(B)

    S = E - C.T @ B_inv @ C
    S = 0.5 * (S + S.T)

    S_inv = recursive_spd_inverse(S)

    top_left = B_inv + B_inv @ C @ S_inv @ C.T @ B_inv
    top_right = -B_inv @ C @ S_inv
    bottom_left = -S_inv @ C.T @ B_inv
    bottom_right = S_inv

    return np.block(
        [
            [top_left, top_right],
            [bottom_left, bottom_right],
        ]
    )


def general_inverse(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=float)

    gram = M.T @ M
    gram_inv = recursive_spd_inverse(gram)

    return gram_inv @ M.T


def incidence_matrix_on_block(
    state: GraphState,
    edges: Sequence[Edge],
    vertices: Sequence[int],
    root: int,
) -> np.ndarray:

    indices = block_indices(vertices, root)

    if not edges:
        return np.zeros((len(indices), 0), dtype=float)

    columns = []

    for edge in edges:
        b = state.reduced_incidence_vector(edge, root)
        columns.append(b[indices])

    return np.column_stack(columns)


def deleted_laplacian_on_block(
    state: GraphState,
    edges: Sequence[Edge],
    vertices: Sequence[int],
    root: int,
) -> np.ndarray:

    indices = block_indices(vertices, root)
    L_D = np.zeros((len(indices), len(indices)), dtype=float)

    for edge in edges:
        b = state.reduced_incidence_vector(edge, root)
        q = b[indices]
        L_D += edge.weight * np.outer(q, q)

    return L_D


def update_block(
    N: np.ndarray,
    state: GraphState,
    vertices: Sequence[int],
    contracted: Sequence[Edge],
    deleted: Sequence[Edge],
    root: int,
) -> None:

    indices = block_indices(vertices, root)

    if not indices:
        return

    block = N[np.ix_(indices, indices)].copy()

    if contracted:
        Q = incidence_matrix_on_block(
            state,
            contracted,
            vertices,
            root,
        )

        middle = Q.T @ block @ Q
        middle = 0.5 * (middle + middle.T)
        middle_inv = recursive_spd_inverse(middle)

        block = (
            block
            - block @ Q @ middle_inv @ Q.T @ block
        )

    if deleted:
        L_D = deleted_laplacian_on_block(
            state,
            deleted,
            vertices,
            root,
        )

        M = L_D @ block - np.eye(len(indices))
        M_inv = general_inverse(M)

        block = block - block @ M_inv @ L_D @ block

    N[np.ix_(indices, indices)] = block


def edge_inclusion_probability(
    N: np.ndarray,
    state: GraphState,
    edge: Edge,
    root: int,
) -> float:

    b = state.reduced_incidence_vector(edge, root)
    resistance = float(b @ N @ b)
    p = edge.weight * resistance

    if p < 0:
        if p > -1e-10:
            p = 0.0
        else:
            raise RuntimeError("Computed a negative edge probability.")

    if p > 1:
        if p < 1.0 + 1e-10:
            p = 1.0
        else:
            raise RuntimeError("Computed an edge probability larger than one.")

    return p


def split_in_half(vertices: Sequence[int]) -> Tuple[List[int], List[int]]:
    vertices = list(vertices)

    if len(vertices) <= 1:
        return vertices, []

    middle = len(vertices) // 2
    return vertices[:middle], vertices[middle:]


def sample_edges_crossing(
    state: GraphState,
    N: np.ndarray,
    R: Sequence[int],
    S: Sequence[int],
    root: int,
    rng: np.random.Generator,
) -> Tuple[List[Edge], List[Edge]]:

    R = list(R)
    S = list(S)

    if not R or not S:
        return [], []

    if len(R) == 1 and len(S) == 1:
        r = R[0]
        s = S[0]
        edge = state.edge_between(r, s)

        if edge is None:
            return [], []

        p = edge_inclusion_probability(N, state, edge, root)

        if rng.random() <= p:
            return [edge], []

        return [], [edge]

    R1, R2 = split_in_half(R)
    S1, S2 = split_in_half(S)

    all_contracted: List[Edge] = []
    all_deleted: List[Edge] = []

    for Ri in (R1, R2):
        for Sj in (S1, S2):
            if not Ri or not Sj:
                continue

            child_vertices = list(Ri) + list(Sj)
            indices = block_indices(child_vertices, root)
            saved = N[np.ix_(indices, indices)].copy()

            contracted, deleted = sample_edges_crossing(
                state,
                N,
                Ri,
                Sj,
                root,
                rng,
            )

            N[np.ix_(indices, indices)] = saved

            update_block(
                N,
                state,
                list(R) + list(S),
                contracted,
                deleted,
                root,
            )

            all_contracted.extend(contracted)
            all_deleted.extend(deleted)

    return all_contracted, all_deleted


def sample_edges_within(
    state: GraphState,
    N: np.ndarray,
    S: Sequence[int],
    root: int,
    rng: np.random.Generator,
) -> Tuple[List[Edge], List[Edge]]:

    S = list(S)

    if len(S) <= 1:
        return [], []

    S1, S2 = split_in_half(S)

    all_contracted: List[Edge] = []
    all_deleted: List[Edge] = []

    for Si in (S1, S2):
        indices = block_indices(Si, root)
        saved = N[np.ix_(indices, indices)].copy()

        contracted, deleted = sample_edges_within(
            state,
            N,
            Si,
            root,
            rng,
        )

        N[np.ix_(indices, indices)] = saved

        update_block(
            N,
            state,
            S,
            contracted,
            deleted,
            root,
        )

        all_contracted.extend(contracted)
        all_deleted.extend(deleted)

    contracted, deleted = sample_edges_crossing(
        state,
        N,
        S1,
        S2,
        root,
        rng,
    )

    all_contracted.extend(contracted)
    all_deleted.extend(deleted)

    return all_contracted, all_deleted


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

    L0 = current.reduced_laplacian(root)
    N = recursive_spd_inverse(L0)

    chosen, _ = sample_edges_within(
        current,
        N,
        list(range(current.n_original)),
        root,
        rng,
    )

    if len(chosen) != state.n_original - 1:
        raise RuntimeError(
            f"Sampler returned {len(chosen)} edges; expected "
            f"{state.n_original - 1}."
        )

    return chosen

