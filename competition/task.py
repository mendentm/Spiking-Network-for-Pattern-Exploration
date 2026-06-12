# competition/task.py
"""
Rate-matched temporal sequence-classification task.
===================================================

Goal (one sentence): produce a labeled set of input spike trains whose classes
are distinguishable *only* by the temporal order of their sub-patterns, not by
average firing rate.

Why rate-matched: if classes differed in mean rate, a memoryless model could win
by counting spikes and the competition would say nothing about topology. Here
every class drives each input group to the same high rate for exactly one epoch,
so the per-channel expected spike count is identical across classes (verified in
`dataset_rate_summary`). Separating the classes therefore requires using *when*
things happened - i.e. temporal integration, which is exactly where recurrent
topology can help and a feedforward network cannot.

Inputs/outputs:
    generate_temporal_dataset(...) -> (X, y, meta)
        X: float32 array [n_samples, N_in, T_task] of binary spikes (0/1)
        y: int array [n_samples] of class labels in [0, n_classes)
        meta: dict describing the task (orders, group map, rates, chance level)
"""

import itertools
import numpy as np


def _make_orders(n_groups, n_classes, rng):
    """Pick `n_classes` distinct group orderings (permutations of the groups)."""
    perms = list(itertools.permutations(range(n_groups)))
    if n_classes > len(perms):
        raise ValueError(
            f"n_classes={n_classes} exceeds the {len(perms)} distinct orderings "
            f"of {n_groups} groups; increase n_groups or lower n_classes."
        )
    rng.shuffle(perms)
    return [list(p) for p in perms[:n_classes]]


def generate_temporal_dataset(n_per_class=150,
                              n_groups=3,
                              n_classes=4,
                              N_in=24,
                              T_task=150,
                              high_rate_hz=80.0,
                              low_rate_hz=10.0,
                              dt_ms=1.0,
                              seed=0):
    """
    Generate the rate-matched temporal classification dataset.

    Args:
        n_per_class:  samples per class.
        n_groups:     number of input channel groups (and number of epochs).
        n_classes:    number of distinct orderings to use as classes (<= n_groups!).
        N_in:         number of input channels (rounded down to a multiple of n_groups).
        T_task:       trial length in timesteps (rounded down to a multiple of n_groups).
        high_rate_hz: Poisson rate of a group during its active epoch.
        low_rate_hz:  baseline Poisson rate otherwise.
        dt_ms:        timestep duration in ms (probability per step = rate * dt / 1000).
        seed:         RNG seed (full determinism).

    Returns:
        (X, y, meta) as described in the module docstring.
    """
    rng = np.random.default_rng(seed)

    group_size = N_in // n_groups
    if group_size < 1:
        raise ValueError(f"N_in={N_in} too small for n_groups={n_groups}.")
    N_in = group_size * n_groups            # enforce divisibility
    epoch_len = T_task // n_groups
    if epoch_len < 1:
        raise ValueError(f"T_task={T_task} too small for n_groups={n_groups}.")
    T_task = epoch_len * n_groups

    orders = _make_orders(n_groups, n_classes, rng)
    p_high = high_rate_hz * dt_ms / 1000.0
    p_low = low_rate_hz * dt_ms / 1000.0

    n_samples = n_per_class * n_classes
    X = np.zeros((n_samples, N_in, T_task), dtype=np.float32)
    y = np.zeros(n_samples, dtype=int)

    idx = 0
    for c in range(n_classes):
        order = orders[c]
        for _ in range(n_per_class):
            rate_map = np.full((N_in, T_task), p_low, dtype=np.float64)
            for position, group in enumerate(order):
                t0 = position * epoch_len
                t1 = t0 + epoch_len
                ch0 = group * group_size
                ch1 = ch0 + group_size
                rate_map[ch0:ch1, t0:t1] = p_high
            X[idx] = (rng.random((N_in, T_task)) < rate_map).astype(np.float32)
            y[idx] = c
            idx += 1

    # Shuffle so classes are interleaved.
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]

    meta = {
        "n_samples": n_samples,
        "n_classes": n_classes,
        "n_groups": n_groups,
        "group_size": group_size,
        "N_in": N_in,
        "T_task": T_task,
        "epoch_len": epoch_len,
        "high_rate_hz": high_rate_hz,
        "low_rate_hz": low_rate_hz,
        "orders": orders,
        "chance_level": 1.0 / n_classes,
        "seed": seed,
    }
    return X, y, meta


def train_test_split(X, y, test_frac=0.25, seed=0):
    """Deterministic stratified-ish split (shuffle then slice)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def dataset_rate_summary(X, y):
    """
    Per-class mean spikes-per-channel-per-trial. These should be nearly identical
    across classes by construction (the rate-matched property). Returned for
    verification/reporting.
    """
    classes = np.unique(y)
    summary = {}
    for c in classes:
        Xc = X[y == c]
        summary[int(c)] = float(Xc.sum(axis=(1, 2)).mean())  # total spikes / trial
    return summary
