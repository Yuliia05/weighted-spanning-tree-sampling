# Weighted Spanning Tree Sampling

Implementations and experiments for exact sampling of weighted spanning trees.

For a connected undirected weighted graph \(G=(V,E,w)\), a spanning tree \(T\) is sampled with probability proportional to

\[
w(T)=\prod_{e\in T} w_e.
\]

The repository contains implementations of four exact sampling approaches and the experiments used to compare them.

## Algorithms

The implementations are located in `algorithms/`:

- `approach1.py` -- deletion-contraction using the Weighted Matrix-Tree Theorem
- `approach2.py` -- effective-resistance probabilities with a maintained Cholesky factorization
- `approach3.py` -- recursive edge processing with inverse Laplacian updates
- `approach4.py` -- weighted Aldous-Broder random-walk algorithm

## Experiments

The experiments are located in `experiments/`:

- `correctness.py` -- compares empirical spanning-tree distributions with the exact distribution
- `runtime.py` -- compares running times on several weighted and unweighted graph families
- `numerical_stability.py` -- studies numerical accuracy and stability of Approaches 1–3

These experiments correspond to Sections 6.1, 6.2, and 6.3 of the paper.

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the experiments

Run the commands from the root directory of the repository.

### Correctness

```bash
python -m experiments.correctness
```

### Running time

```bash
python -m experiments.runtime
```

### Numerical stability

```bash
python -m experiments.numerical_stability
```

The generated CSV files are saved in the `experiments/` directory.

## Author

Yuliia Tatarinova

Supervisor: Katarzyna Grygiel