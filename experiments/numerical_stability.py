import csv
from pathlib import Path

import mpmath as mp
import numpy as np

from algorithms import approach1
from algorithms import approach2
from algorithms import approach3


n = 30
root = 0

q_values = [0, 2, 4, 6, 8, 10, 12]

repetitions = 3
graph_seed = 20260829
sampler_seed = 913700


def make_base_graph():
    rng = np.random.default_rng(graph_seed)
    edge_set = set()

    for u in range(n):
        v = (u + 1) % n
        edge_set.add(tuple(sorted((u, v))))

    # Add random edges.
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in edge_set:
                continue

            if rng.random() < 0.15:
                edge_set.add((u, v))

    pairs = sorted(edge_set)
    exponents = rng.uniform(-1.0, 1.0, len(pairs))

    return [
        (u, v, float(x))
        for (u, v), x in zip(pairs, exponents)
    ]


def set_weights(base_edges, q):
    return [
        (u, v, 10.0 ** (q * x))
        for u, v, x in base_edges
    ]


def reduced_laplacian(edges):
    L = np.zeros((n, n))

    for u, v, weight in edges:
        L[u, u] += weight
        L[v, v] += weight
        L[u, v] -= weight
        L[v, u] -= weight

    return L[1:, 1:]


def factor_residual(A, U):
    if A.size == 0:
        return 0.0

    return (
        np.linalg.norm(U.T @ U - A, ord="fro")
        / np.linalg.norm(A, ord="fro")
    )


def inverse_residual(A, A_inv):
    I = np.eye(len(A))

    return (
        np.linalg.norm(A @ A_inv - I, ord="fro")
        / np.linalg.norm(I, ord="fro")
    )


def high_precision_probabilities(edges):
    mp.mp.dps = 100

    A = mp.matrix(n - 1, n - 1)

    for u, v, weight in edges:
        w = mp.mpf(str(weight))

        if u != root:
            i = u - 1
            A[i, i] += w

        if v != root:
            j = v - 1
            A[j, j] += w

        if u != root and v != root:
            i = u - 1
            j = v - 1
            A[i, j] -= w
            A[j, i] -= w

    A_inv = A ** -1
    probabilities = []

    for u, v, weight in edges:
        b = mp.matrix(n - 1, 1)

        if u != root:
            b[u - 1] += 1

        if v != root:
            b[v - 1] -= 1

        resistance = (b.T * A_inv * b)[0]
        p = mp.mpf(str(weight)) * resistance

        probabilities.append(float(p))

    return probabilities


def max_probability_error(probabilities, reference):
    if probabilities is None:
        return None

    if not all(np.isfinite(p) for p in probabilities):
        return None

    return max(
        abs(p - p_ref)
        for p, p_ref in zip(probabilities, reference)
    )


def invalid_probability_count(probabilities):
    if probabilities is None:
        return ""

    return sum(
        not np.isfinite(p) or p < 0 or p > 1
        for p in probabilities
    )


def approach1_probabilities(state):
    probabilities = []

    for edge in state.edges:
        deleted = state.copy()
        deleted.delete(edge)
        tau_delete = approach1.tau_weighted(deleted)

        contracted = state.copy()
        contracted.contract(edge)
        tau_contract = approach1.tau_weighted(contracted)

        numerator = edge.weight * tau_contract
        denominator = tau_delete + numerator

        if (
            not np.isfinite(tau_delete)
            or not np.isfinite(tau_contract)
            or not np.isfinite(denominator)
            or denominator == 0
        ):
            return None

        probabilities.append(
            float(numerator / denominator)
        )

    return probabilities


def approach2_initial_data(state):
    A = state.reduced_laplacian(root)
    U = approach2.cholesky_factorization(A)

    residual = factor_residual(A, U)
    probabilities = []

    for edge in state.edges:
        b = state.incidence_vector(edge, root)
        y = approach2.forward_substitution(U, b)

        probabilities.append(
            float(edge.weight * np.dot(y, y))
        )

    return residual, probabilities


def approach3_initial_data(state):
    A = state.reduced_laplacian(root)
    A_inv = approach3.recursive_spd_inverse(A)

    residual = inverse_residual(A, A_inv)
    probabilities = []

    for edge in state.edges:
        b = state.reduced_incidence_vector(edge, root)

        probabilities.append(
            float(edge.weight * (b @ A_inv @ b))
        )

    return residual, probabilities


def run_approach1(state, seed):
    rng = np.random.default_rng(seed)

    tree = approach1.sample_weighted_spanning_tree(
        state,
        rng=rng,
    )

    if len(tree) != n - 1:
        raise RuntimeError("The result is not a spanning tree")


def run_approach2(state, seed):
    rng = np.random.default_rng(seed)

    current = state.copy()
    current_root = root
    chosen = []

    A = current.reduced_laplacian(current_root)
    U = approach2.cholesky_factorization(A)

    max_residual = 0.0

    while current.num_nodes() > 1:
        edge = current.next_edge()

        if edge is None:
            raise RuntimeError("No edge remains")

        u = current.supernode(edge.u)
        v = current.supernode(edge.v)

        b = current.incidence_vector(
            edge,
            current_root,
        )

        p = approach2.edge_inclusion_probability(
            U,
            b,
            edge.weight,
        )

        if rng.random() <= p:
            chosen.append(edge)

            new_U = approach2.contract_factor(
                U,
                current,
                u,
                v,
                current_root,
            )

            contracted = current.copy()
            new_root = contracted.contract(
                edge,
                current_root,
            )

            target = contracted.reduced_laplacian(
                new_root
            )

            residual = factor_residual(
                target,
                new_U,
            )

            U = new_U
            current_root = current.contract(
                edge,
                current_root,
            )

        else:
            x = np.sqrt(edge.weight) * b

            new_U = approach2.chol_downdate(
                U,
                x,
            )

            target = U.T @ U - np.outer(x, x)

            residual = factor_residual(
                target,
                new_U,
            )

            U = new_U
            current.delete(edge)

        max_residual = max(
            max_residual,
            float(residual),
        )

    if len(chosen) != n - 1:
        raise RuntimeError("The result is not a spanning tree")

    return max_residual


def run_approach3(state, seed):
    rng = np.random.default_rng(seed)

    tree = approach3.sample_weighted_spanning_tree(
        state,
        root=root,
        rng=rng,
    )

    if len(tree) != n - 1:
        raise RuntimeError("The result is not a spanning tree")


def main():
    base_edges = make_base_graph()

    print(f"n={n}, m={len(base_edges)}")

    summary = []
    runs = []

    for q_index, q in enumerate(q_values):
        edges = set_weights(base_edges, q)
        L0 = reduced_laplacian(edges)

        condition_number = np.linalg.cond(L0)

        print(
            f"\nq={q}, "
            f"condition number={condition_number:.3e}"
        )

        reference = high_precision_probabilities(edges)

        # Approach 1

        try:
            state1 = approach1.make_graph(n, edges)
            probabilities1 = approach1_probabilities(state1)

            error1 = max_probability_error(
                probabilities1,
                reference,
            )
            initial_error1 = ""

        except Exception as exc:
            state1 = None
            probabilities1 = None
            error1 = None
            initial_error1 = str(exc)

        success1 = 0

        for rep in range(repetitions):
            seed = (
                sampler_seed
                + 10000 * q_index
                + rep
            )

            if state1 is None:
                status = "error"
                error = initial_error1

            else:
                try:
                    run_approach1(state1, seed)
                    status = "ok"
                    error = ""
                    success1 += 1

                except Exception as exc:
                    status = "error"
                    error = str(exc)

            runs.append({
                "q": q,
                "approach": "Approach 1",
                "run": rep + 1,
                "status": status,
                "max factor residual": "",
                "error": error,
            })

        # Approach 2

        state2 = approach2.make_graph(n, edges)

        residual2, probabilities2 = (
            approach2_initial_data(state2)
        )

        error2 = max_probability_error(
            probabilities2,
            reference,
        )

        success2 = 0
        update_residuals = []

        for rep in range(repetitions):
            seed = (
                sampler_seed
                + 10000 * q_index
                + 100
                + rep
            )

            try:
                residual = run_approach2(
                    state2,
                    seed,
                )

                status = "ok"
                error = ""
                success2 += 1
                update_residuals.append(residual)

            except Exception as exc:
                status = "error"
                error = str(exc)
                residual = ""

            runs.append({
                "q": q,
                "approach": "Approach 2",
                "run": rep + 1,
                "status": status,
                "max factor residual": residual,
                "error": error,
            })

        if update_residuals:
            max_update_residual = max(update_residuals)
        else:
            max_update_residual = ""

        # Approach 3

        state3 = approach3.make_graph(n, edges)

        residual3, probabilities3 = (
            approach3_initial_data(state3)
        )

        error3 = max_probability_error(
            probabilities3,
            reference,
        )

        success3 = 0

        for rep in range(repetitions):
            seed = (
                sampler_seed
                + 10000 * q_index
                + 200
                + rep
            )

            try:
                run_approach3(state3, seed)
                status = "ok"
                error = ""
                success3 += 1

            except Exception as exc:
                status = "error"
                error = str(exc)

            runs.append({
                "q": q,
                "approach": "Approach 3",
                "run": rep + 1,
                "status": status,
                "max factor residual": "",
                "error": error,
            })

        summary.append({
            "q": q,
            "n": n,
            "m": len(edges),
            "condition number": condition_number,

            "A1 probability error":
                "overflow" if error1 is None else error1,
            "A1 successful runs": success1,
            "A1 invalid initial probabilities":
                invalid_probability_count(probabilities1),

            "A2 probability error": error2,
            "A2 successful runs": success2,
            "A2 initial residual": residual2,
            "A2 max update residual": max_update_residual,
            "A2 invalid initial probabilities":
                invalid_probability_count(probabilities2),

            "A3 probability error": error3,
            "A3 successful runs": success3,
            "A3 initial residual": residual3,
            "A3 invalid initial probabilities":
                invalid_probability_count(probabilities3),
        })

        print(
            f"  A1: "
            f"{'overflow' if error1 is None else f'{error1:.3e}'}, "
            f"{success1}/3"
        )

        print(
            f"  A2: {error2:.3e}, "
            f"{success2}/3"
        )

        print(
            f"  A3: {error3:.3e}, "
            f"{success3}/3"
        )

    summary_file = Path(__file__).with_name(
        "numerical_stability.csv"
    )

    runs_file = Path(__file__).with_name(
        "numerical_stability_runs.csv"
    )

    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary)

    with runs_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=runs[0].keys(),
        )
        writer.writeheader()
        writer.writerows(runs)


if __name__ == "__main__":
    main()