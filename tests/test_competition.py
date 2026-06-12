# tests/test_competition.py
"""
Smoke tests for the competition subsystem and the patched generators.

Runnable two ways:
    pytest tests/test_competition.py
    python tests/test_competition.py        (no pytest required)

These are fast (tiny config) sanity checks, not a full benchmark: they verify the
pipeline runs end to end, outputs are well formed, results are deterministic, the
rate-matched task and reservoir behave as designed, and the topology machinery
(adjacency generators + firing-rate calibration) works.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from competition.task import generate_temporal_dataset, dataset_rate_summary
from competition.reservoir import IzhikevichReservoir
from competition.connectivity import make_adjacency, TOPOLOGIES
from competition.compete import run_competition, FFSNN, MLP, _rsnn_name


_TINY = dict(
    n_per_class=40, n_groups=3, n_classes=3, N_in=15, T_task=60,
    high_rate_hz=120.0, reservoir_N=80, topologies=["random", "smallworld"],
    include_feedforward=True, match_firing_hz=None, rec_scale=14.0,
    conn_prob=0.12, mlp_epochs=40, seeds=[0, 1], make_plot=False, verbose=False,
)


def _contestants():
    return [_rsnn_name(t) for t in _TINY["topologies"]] + [FFSNN, MLP]


def test_task_is_rate_matched():
    """Per-class mean spike counts must be near-identical (the rate-match property)."""
    X, y, meta = generate_temporal_dataset(n_per_class=80, seed=0)
    counts = np.array(list(dataset_rate_summary(X, y).values()))
    spread = counts.max() - counts.min()
    assert spread / counts.mean() < 0.10, f"classes not rate-matched (spread={spread})"


def test_reservoir_fires_and_is_finite():
    """Reservoir should fire (not silent, not saturated) and produce finite features."""
    X, _, meta = generate_temporal_dataset(n_per_class=10, seed=0)
    res = IzhikevichReservoir(N=100, N_in=meta["N_in"], recurrent=True, seed=0)
    feats, hz = res.transform(X[:20], return_rates_hz=True)
    assert np.isfinite(feats).all()
    assert 1.0 < hz < 200.0, f"unhealthy firing rate {hz} Hz"


def test_feedforward_differs_from_recurrent():
    """recurrent=False must actually change the dynamics vs recurrent=True."""
    X, _, meta = generate_temporal_dataset(n_per_class=10, seed=0)
    rec = IzhikevichReservoir(N=120, N_in=meta["N_in"], recurrent=True, seed=0).transform(X[:20])
    ff = IzhikevichReservoir(N=120, N_in=meta["N_in"], recurrent=False, seed=0).transform(X[:20])
    assert not np.allclose(rec, ff), "recurrent and feedforward produced identical features"


def test_topologies_valid():
    """Every topology generator yields no self-loops and a plausible density."""
    N, p = 200, 0.1
    for t in TOPOLOGIES:
        A = make_adjacency(t, N, p, np.random.default_rng(1))
        assert A.shape == (N, N)
        assert not np.any(np.diag(A)), f"{t} has self-loops"
        d = A.mean()
        assert 0.3 * p < d < 3.0 * p, f"{t} density {d:.3f} far from target {p}"


def test_topologies_share_neurons_and_input():
    """For a given seed, only the recurrent wiring should differ across topologies."""
    r1 = IzhikevichReservoir(N=120, N_in=15, topology="random", seed=0)
    r2 = IzhikevichReservoir(N=120, N_in=15, topology="smallworld", seed=0)
    assert np.allclose(r1.W_in, r2.W_in), "input projection differs across topologies"
    assert np.allclose(r1.a, r2.a) and np.allclose(r1.c, r2.c), "neurons differ across topologies"
    assert not np.allclose(r1.W_rec, r2.W_rec), "recurrent wiring should differ"


def test_calibration_controls_firing_rate():
    """Calibrating to a high target yields more spiking than a low target."""
    X, _, meta = generate_temporal_dataset(n_per_class=10, seed=0)
    hi = IzhikevichReservoir(N=150, N_in=meta["N_in"], recurrent=True, seed=0)
    hi.calibrate_rec_scale(X[:20], target_hz=60.0)
    _, hi_hz = hi.transform(X[:20], return_rates_hz=True)
    lo = IzhikevichReservoir(N=150, N_in=meta["N_in"], recurrent=True, seed=0)
    lo.calibrate_rec_scale(X[:20], target_hz=10.0)
    _, lo_hz = lo.transform(X[:20], return_rates_hz=True)
    assert hi_hz > lo_hz, f"calibration did not raise firing (hi={hi_hz}, lo={lo_hz})"


def test_competition_runs_and_is_well_formed():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {**_TINY, "output_dir": tmp}
        res = run_competition(cfg)

        chance = res["chance_level"]
        for name in _contestants():
            acc = res["accuracy"][name]["mean"]
            assert 0.0 <= acc <= 1.0, f"{name} accuracy out of range: {acc}"
        # The task is learnable from the raw input: the MLP must solve it (validates
        # labels/encoding). At tiny scale the reservoirs can sit at chance with a
        # 30-sample test set, so their accuracy is not a reliable smoke signal here -
        # the full run (run_competition.py) is where the topology effect appears.
        assert res["accuracy"][MLP]["mean"] > chance + 0.3, "task not learnable - check labels"
        # Topology bookkeeping is present and self-consistent.
        assert len(res["topology_ranking"]) == len(_TINY["topologies"])
        assert res["best_vs_worst"]["n"] == len(_TINY["seeds"])
        assert any(f.endswith(".json") for f in os.listdir(tmp))


def test_determinism():
    """Same config + seeds => identical accuracies."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {**_TINY, "output_dir": tmp}
        r1 = run_competition(cfg)
        r2 = run_competition(cfg)
        for name in _contestants():
            assert r1["accuracy"][name]["runs"] == r2["accuracy"][name]["runs"], \
                f"{name} not deterministic"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
