# tests/test_evolution.py
"""
Smoke tests for the multi-objective evolutionary engine and its two problems.

Runnable two ways:
    pytest tests/test_evolution.py
    python tests/test_evolution.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from competition.evolution import (non_dominated_sort, crowding_distance,
                                    run_evolution, NetworkProblem, PatternProblem)


def test_non_dominated_sort():
    # Maximization: (1,0),(0,1),(0.5,0.5) are mutually non-dominated; (0,0) is dominated.
    M = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    fronts, rank = non_dominated_sort(M)
    assert set(fronts[0]) == {0, 1, 2}, f"front 0 wrong: {fronts[0]}"
    assert fronts[1] == [3]
    assert rank[3] == 1


def test_crowding_distance_endpoints_infinite():
    M = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    d = crowding_distance(M)
    assert np.isinf(d).sum() >= 2          # boundary solutions get infinite crowding
    assert np.isfinite(d[1])               # interior solution is finite


def _tiny(target, **over):
    cfg = dict(target=target, pop_size=6, generations=2, output_dir=None,
               make_plot=False, verbose=False)
    if target == "network":
        cfg.update(eval_n_per_class=30, eval_T=80)
    else:
        cfg.update(pattern_T=500)
    cfg.update(over)
    return cfg


def test_network_evolution_runs():
    with tempfile.TemporaryDirectory() as tmp:
        res = run_evolution(_tiny("network", output_dir=tmp))
        front = res["pareto_front"]
        assert len(front) >= 1
        for p in front:
            assert 0.0 <= p["objectives"]["accuracy"] <= 1.0
            assert p["objectives"]["energy_nj"] > 0
            assert p["genome"]["topology"] in NetworkProblem.categoricals["topology"]
        assert any(f.endswith(".json") for f in os.listdir(tmp))


def test_pattern_evolution_runs():
    with tempfile.TemporaryDirectory() as tmp:
        res = run_evolution(_tiny("pattern", output_dir=tmp))
        front = res["pareto_front"]
        assert len(front) >= 1
        names = PatternProblem.categoricals["pattern_name"]
        for p in front:
            assert p["genome"]["pattern_name"] in names


def test_pareto_front_is_non_dominated():
    """No solution on the returned front may dominate another on the front."""
    with tempfile.TemporaryDirectory() as tmp:
        res = run_evolution(_tiny("network", output_dir=tmp))
        front = res["pareto_front"]
        pts = np.array([[p["objectives"]["accuracy"], -p["objectives"]["energy_nj"]]
                        for p in front])
        for i in range(len(pts)):
            for j in range(len(pts)):
                if i == j:
                    continue
                dominates = np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i])
                assert not dominates, "front contains a dominated solution"


def test_determinism():
    with tempfile.TemporaryDirectory() as tmp:
        r1 = run_evolution(_tiny("network", output_dir=tmp))
        r2 = run_evolution(_tiny("network", output_dir=tmp))
        f1 = sorted(p["objectives"]["accuracy"] for p in r1["pareto_front"])
        f2 = sorted(p["objectives"]["accuracy"] for p in r2["pareto_front"])
        assert f1 == f2, "evolution not deterministic for a fixed seed"


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
