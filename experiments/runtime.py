import csv
import math
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import numpy as np

from algorithms import approach1
from algorithms import approach2
from algorithms import approach3
from algorithms import approach4


repetitions = 3
timeout = 300


approaches = [
    ("Approach 1", approach1),
    ("Approach 2", approach2),
    ("Approach 3", approach3),
    ("Approach 4", approach4),
]


sizes = {
    "Complete": [10, 20, 50, 100, 200, 500, 1000],
    "Sparse unweighted": [
        10, 20, 50, 100, 200, 500, 1000, 5000, 10000
    ],
    "Dense weighted": [10, 20, 50, 100, 200, 500, 1000],
    "Lollipop": [10, 20, 50, 100, 200, 500, 1000],
    "Weighted weak cut": [10, 20, 50, 100],
}


def is_connected(n, edges):
    neighbours = [[] for _ in range(n)]

    for u, v, _ in edges:
        neighbours[u].append(v)
        neighbours[v].append(u)

    visited = {0}
    stack = [0]

    while stack:
        v = stack.pop()

        for u in neighbours[v]:
            if u not in visited:
                visited.add(u)
                stack.append(u)

    return len(visited) == n


def complete_graph(n):
    return [
        (u, v, 1.0)
        for u in range(n)
        for v in range(u + 1, n)
    ]


def sparse_graph(n):
    p = 3 * math.log(n) / n
    rng = np.random.default_rng(720000 + n)

    while True:
        edges = []

        for u in range(n - 1):
            mask = rng.random(n - u - 1) < p

            for offset in np.flatnonzero(mask):
                v = u + 1 + int(offset)
                edges.append((u, v, 1.0))

        if is_connected(n, edges):
            return edges


def dense_weighted_graph(n):
    rng = np.random.default_rng(750000 + n)

    while True:
        edges = []

        for u in range(n - 1):
            mask = rng.random(n - u - 1) < 0.5
            vertices = np.flatnonzero(mask) + u + 1
            weights = rng.uniform(1.0, 10.0, len(vertices))

            for v, weight in zip(vertices, weights):
                edges.append((u, int(v), float(weight)))

        if is_connected(n, edges):
            return edges


def lollipop_graph(n):
    k = 2 * n // 3
    edges = []

    for u in range(k):
        for v in range(u + 1, k):
            edges.append((u, v, 1.0))

    for v in range(k, n):
        edges.append((v - 1, v, 1.0))

    return edges


def weak_cut_graph(n):
    k = n // 2
    edges = []

    for u in range(k):
        for v in range(u + 1, k):
            edges.append((u, v, 1.0))

    for u in range(k, n):
        for v in range(u + 1, n):
            edges.append((u, v, 1.0))

    edges.append((0, k, 5e-6))
    edges.append((1, k + 1, 5e-6))

    return edges


def make_graph(family, n):
    if family == "Complete":
        return complete_graph(n)

    if family == "Sparse unweighted":
        return sparse_graph(n)

    if family == "Dense weighted":
        return dense_weighted_graph(n)

    if family == "Lollipop":
        return lollipop_graph(n)

    if family == "Weighted weak cut":
        return weak_cut_graph(n)


def sample_seed(family, n, approach, rep):
    if family == "Complete":
        return 900000 + 1000 * n + 100 * approach + rep

    if family == "Sparse unweighted":
        return 1100000 + 1000 * n + 100 * approach + rep

    if family == "Dense weighted":
        return 1400000 + 1000 * n + 100 * approach + rep

    if family == "Lollipop":
        return 1600000 + 1000 * n + 100 * approach + rep

    i = [10, 20, 50, 100].index(n)

    return 123456 + 10000 * i + 100 * approach + rep


def run_once(approach_number, n, edges, seed):
    name, module = approaches[approach_number]
    rng = np.random.default_rng(seed)

    start = time.perf_counter()

    graph = module.make_graph(n, edges)

    if name == "Approach 1":
        tree = module.sample_weighted_spanning_tree(
            graph,
            rng=rng,
        )
        steps = None

    elif name in ("Approach 2", "Approach 3"):
        tree = module.sample_weighted_spanning_tree(
            graph,
            root=0,
            rng=rng,
        )
        steps = None

    else:
        tree, steps = module.sample_weighted_spanning_tree(
            graph,
            start_vertex=0,
            rng=rng,
            return_steps=True,
        )

    elapsed = time.perf_counter() - start

    if len(tree) != n - 1:
        raise RuntimeError("The result is not a spanning tree")

    return elapsed, steps


def worker(connection, approach_number, n, edges, seed):
    try:
        elapsed, steps = run_once(
            approach_number,
            n,
            edges,
            seed,
        )

        connection.send(("ok", elapsed, steps, ""))

    except Exception as error:
        connection.send(("error", None, None, str(error)))

    connection.close()


def run_with_timeout(approach_number, n, edges, seed):
    parent, child = mp.Pipe()

    process = mp.Process(
        target=worker,
        args=(child, approach_number, n, edges, seed),
    )

    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()

        return "timeout", None, None, ""

    if parent.poll():
        return parent.recv()

    return "error", None, None, "No result returned"


def main():
    runs = []
    summary = []

    stopped = {
        family: set()
        for family in sizes
    }

    for family in sizes:
        print(f"\n{family}")

        for n in sizes[family]:
            edges = make_graph(family, n)

            print(f"\nn={n}, m={len(edges)}")

            for approach_number, (name, _) in enumerate(approaches):

                if approach_number in stopped[family]:
                    print(f"  {name}: skipped")

                    summary.append({
                        "family": family,
                        "n": n,
                        "approach": name,
                        "median time": "",
                        "median steps": "",
                        "status": "skipped",
                    })

                    continue

                times = []
                steps = []
                status = "ok"

                for rep in range(repetitions):
                    seed = sample_seed(
                        family,
                        n,
                        approach_number,
                        rep,
                    )

                    status, elapsed, walk_steps, error = run_with_timeout(
                        approach_number,
                        n,
                        edges,
                        seed,
                    )

                    runs.append({
                        "family": family,
                        "n": n,
                        "approach": name,
                        "run": rep + 1,
                        "time": elapsed if elapsed is not None else "",
                        "steps": walk_steps if walk_steps is not None else "",
                        "status": status,
                        "error": error,
                    })

                    if status == "timeout":
                        print(f"  {name}, run {rep + 1}: timeout")
                        stopped[family].add(approach_number)
                        break

                    if status == "error":
                        print(f"  {name}, run {rep + 1}: error")
                        print(f"    {error}")
                        break

                    times.append(elapsed)

                    if walk_steps is not None:
                        steps.append(walk_steps)
                        print(
                            f"  {name}, run {rep + 1}: "
                            f"{elapsed:.6f} s, "
                            f"{walk_steps} steps"
                        )
                    else:
                        print(
                            f"  {name}, run {rep + 1}: "
                            f"{elapsed:.6f} s"
                        )

                if status == "ok":
                    median_time = statistics.median(times)

                    if steps:
                        median_steps = statistics.median(steps)
                    else:
                        median_steps = ""

                    print(
                        f"    median time: "
                        f"{median_time:.6f} s"
                    )

                    if steps:
                        print(
                            f"    median steps: "
                            f"{median_steps}"
                        )

                else:
                    median_time = ""
                    median_steps = ""

                summary.append({
                    "family": family,
                    "n": n,
                    "approach": name,
                    "median time": median_time,
                    "median steps": median_steps,
                    "status": status,
                })

    runs_file = Path(__file__).with_name("runtime_runs.csv")
    summary_file = Path(__file__).with_name("runtime_summary.csv")

    with runs_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=runs[0].keys(),
        )
        writer.writeheader()
        writer.writerows(runs)

    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()