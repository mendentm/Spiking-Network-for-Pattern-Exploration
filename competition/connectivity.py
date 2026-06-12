# competition/connectivity.py
"""
Network topology generators for the reservoir.
==============================================

Each generator returns a boolean adjacency matrix A of shape [N, N] (no
self-loops) with approximately the requested connection density, where
A[i, j] = True means a synapse from presynaptic neuron j to postsynaptic i.

The point of the competition is to vary *only* this structure (at matched
density, and - in `compete.py` - matched firing rate) and ask whether it
changes how well the spiking reservoir computes.

Topologies
----------
random      Erdos-Renyi: each edge independent with prob = density. The
            unstructured baseline.
ring        1D ring lattice: each neuron connects to its nearest neighbors.
            High clustering, long path length, spatially local.
smallworld  Watts-Strogatz: a ring with a fraction of edges randomly rewired.
            High clustering *and* short path length (the "small-world" regime).
modular     Block/community structure: dense within modules, sparse between.
            Fragments the network into weakly-coupled communities.
scalefree   Barabasi-Albert preferential attachment: a few high-degree hubs.
"""

import numpy as np

TOPOLOGIES = ["random", "ring", "smallworld", "modular", "scalefree"]


def _random(N, density, rng):
    A = rng.random((N, N)) < density
    np.fill_diagonal(A, False)
    return A


def _ring(N, density, rng):
    k = max(1, int(round(density * N / 2)))      # neighbors on each side
    A = np.zeros((N, N), dtype=bool)
    idx = np.arange(N)
    for d in range(1, k + 1):
        A[idx, (idx + d) % N] = True
        A[idx, (idx - d) % N] = True
    return A


def _smallworld(N, density, rng, beta=0.3):
    A = _ring(N, density, rng)
    src, dst = np.where(A)
    for s, d in zip(src, dst):
        if rng.random() < beta:
            A[s, d] = False
            nd = int(rng.integers(N))
            while nd == s or A[s, nd]:
                nd = int(rng.integers(N))
            A[s, nd] = True
    return A


def _modular(N, density, rng, n_modules=5, ratio=10.0):
    """Dense within modules, sparse between; overall density ~= `density`."""
    mod = np.arange(N) % n_modules
    same = mod[:, None] == mod[None, :]
    # Solve for p_out so the overall mean degree matches `density`:
    #   density = p_out * (ratio/M + (M-1)/M)
    p_out = density * n_modules / (ratio + n_modules - 1)
    P = np.where(same, ratio * p_out, p_out)
    A = rng.random((N, N)) < P
    np.fill_diagonal(A, False)
    return A


def _scalefree(N, density, rng):
    """Undirected preferential attachment (Barabasi-Albert style)."""
    m = max(1, int(round(density * N / 2)))       # edges added per new node
    A = np.zeros((N, N), dtype=bool)
    deg = np.zeros(N)
    init = m + 1
    A[:init, :init] = True
    np.fill_diagonal(A, False)
    deg[:init] = init - 1
    for nw in range(init, N):
        probs = deg[:nw] / deg[:nw].sum()
        targets = rng.choice(nw, size=min(m, nw), replace=False, p=probs)
        for t in targets:
            A[nw, t] = A[t, nw] = True
            deg[nw] += 1
            deg[t] += 1
    np.fill_diagonal(A, False)
    return A


_GENERATORS = {
    "random": _random,
    "ring": _ring,
    "smallworld": _smallworld,
    "modular": _modular,
    "scalefree": _scalefree,
}


def make_adjacency(topology, N, density, rng):
    """Build a boolean [N, N] adjacency for the named topology."""
    if topology not in _GENERATORS:
        raise ValueError(f"Unknown topology '{topology}'. Choose from {TOPOLOGIES}.")
    return _GENERATORS[topology](N, density, rng)
