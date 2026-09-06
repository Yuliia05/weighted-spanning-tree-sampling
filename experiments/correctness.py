import csv
import itertools
from collections import Counter
from pathlib import Path

import numpy as np


from algorithms import approach1
from algorithms import approach2
from algorithms import approach3
from algorithms import approach4


samples = 10000
repetitions = 30


graphs = [
    (
        "Triangle",
        3,
        [
            (0, 1, 1),
            (0, 2, 2),
            (1, 2, 4),
        ],
    ),
    (
        "Weighted K4",
        4,
        [
            (0, 1, 1),
            (0, 2, 2),
            (0, 3, 3),
            (1, 2, 5),
            (1, 3, 7),
            (2, 3, 10),
        ],
    ),
    (
        "Sparse, n=5",
        5,
        [
            (0, 1, 2),
            (1, 2, 9),
            (2, 3, 4),
            (3, 4, 10),
            (0, 4, 3),
            (0, 2, 7),
            (1, 4, 6),
        ],
    ),
    (
        "Weighted K5",
        5,
        [
            (0, 1, 2),
            (0, 2, 9),
            (0, 3, 4),
            (0, 4, 10),
            (1, 2, 3),
            (1, 3, 7),
            (1, 4, 6),
            (2, 3, 8),
            (2, 4, 1),
            (3, 4, 5),
        ],
    ),
]


def is_tree(n, edges, chosen):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a = find(a)
        b = find(b)

        if a == b:
            return False

        parent[b] = a
        return True

    for i in chosen:
        u, v, _ = edges[i]
        if not union(u, v):
            return False

    root = find(0)
    return all(find(v) == root for v in range(n))


def exact_distribution(n, edges):
    weights = {}

    for chosen in itertools.combinations(range(len(edges)), n - 1):
        if not is_tree(n, edges, chosen):
            continue

        weight = 1
        for i in chosen:
            weight *= edges[i][2]

        weights[chosen] = weight

    total = sum(weights.values())

    return {
        tree: weight / total
        for tree, weight in weights.items()
    }


def sample_distribution(module, name, n, edges, seed):
    graph = module.make_graph(n, edges)
    rng = np.random.default_rng(seed)
    counts = Counter()

    for _ in range(samples):
        if name in ("approach2", "approach3"):
            tree = module.sample_weighted_spanning_tree(
                graph, root=0, rng=rng
            )
        elif name == "approach4":
            tree = module.sample_weighted_spanning_tree(
                graph, start_vertex=0, rng=rng
            )
        else:
            tree = module.sample_weighted_spanning_tree(
                graph, rng=rng
            )

        ids = tuple(sorted(edge.id for edge in tree))
        counts[ids] += 1

    return {
        tree: count / samples
        for tree, count in counts.items()
    }


def tv_distance(p, q):
    trees = set(p) | set(q)

    return 0.5 * sum(
        abs(p.get(tree, 0) - q.get(tree, 0))
        for tree in trees
    )


approaches = [
    ("Approach 1", "approach1", approach1),
    ("Approach 2", "approach2", approach2),
    ("Approach 3", "approach3", approach3),
    ("Approach 4", "approach4", approach4),
]


def main():
    rows = []
    k5_intervals = []

    for graph_number, (graph_name, n, edges) in enumerate(graphs):
        exact = exact_distribution(n, edges)

        probabilities = np.array(list(exact.values()))
        rng = np.random.default_rng(900000 + 1000 * graph_number)

        baseline = []

        for _ in range(repetitions):
            counts = rng.multinomial(samples, probabilities)
            empirical = counts / samples
            baseline.append(
                0.5 * np.abs(empirical - probabilities).sum()
            )

        result = {
            "graph": graph_name,
            "number of spanning trees": len(exact),
            "exact sampler": np.median(baseline),
        }

        if graph_name == "Weighted K5":
            k5_intervals.append({
                "approach": "Exact sampler",
                "5th percentile": np.quantile(baseline, 0.05),
                "95th percentile": np.quantile(baseline, 0.95),
            })

        for approach_number, (label, name, module) in enumerate(approaches):
            distances = []

            for rep in range(repetitions):
                seed = (
                    10000
                    + 100000 * graph_number
                    + 1000 * approach_number
                    + rep
                )

                empirical = sample_distribution(
                    module, name, n, edges, seed
                )

                distances.append(
                    tv_distance(exact, empirical)
                )

            result[label] = np.median(distances)

            if graph_name == "Weighted K5":
                k5_intervals.append({
                    "approach": label,
                    "5th percentile": np.quantile(distances, 0.05),
                    "95th percentile": np.quantile(distances, 0.95),
                })

        rows.append(result)

    for row in rows:
        print(row)

    print("\nWeighted K5 5th-95th percentile intervals:")
    for row in k5_intervals:
        print(row)

    output = Path(__file__).with_name("correctness.csv")

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    intervals_output = Path(__file__).with_name(
        "correctness_k5_intervals.csv"
    )

    with intervals_output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=k5_intervals[0].keys(),
        )
        writer.writeheader()
        writer.writerows(k5_intervals)


if __name__ == "__main__":
    main()
