# competition/evolution.py
"""
Multi-objective evolutionary engine (NSGA-II-lite) + two problems.
==================================================================

A single, shared evolutionary engine that optimizes a *Pareto front* over two
competing objectives, applied to either of two targets:

- NetworkProblem  - evolve a spiking reservoir (topology + hyperparameters) to
                    trade off task **accuracy** (max) against estimated
                    **energy** (min). Reuses the competition task, reservoir, and
                    energy model. "Does evolution rediscover good topologies?"
- PatternProblem  - evolve a synthetic pattern generator (which generator + its
                    parameters) to trade off two descriptive metrics from
                    metrics.py (default: synchrony_index vs wave_score). Extends
                    the project's original "autonomous pattern exploration".

The engine is generic: a Problem declares its categorical/numeric genes, its two
objectives (name + 'max'/'min'), and an evaluate() method; the engine handles
random init, tournament selection on (Pareto rank, crowding distance), uniform
crossover, mutation, and (mu+lambda) elitist survival - i.e. NSGA-II.

Entry point: `run_evolution(config)` (see `run_evolution.py` at the repo root).
"""

import os
import sys
import json
import inspect
from datetime import datetime

import numpy as np

from .task import generate_temporal_dataset, train_test_split
from .reservoir import IzhikevichReservoir
from .classifiers import RidgeReadout
from .connectivity import TOPOLOGIES
from .compete import _snn_energy_nj

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patterns import (
    generate_traveling_wave, generate_synchronized_bursts, generate_spiral_wave,
    generate_random_activity, generate_localized_clusters, generate_ripple_pattern,
    generate_checkerboard_pattern, generate_game_of_life,
)
from metrics import calculate_all_metrics


# ============================================================================
# NSGA-II core (operates on a maximization objective matrix)
# ============================================================================

def _dominates(a, b):
    """True if a Pareto-dominates b (both are maximization vectors)."""
    return np.all(a >= b) and np.any(a > b)


def non_dominated_sort(M):
    """Sort rows of maximization matrix M [n, k] into Pareto fronts.

    Returns (fronts, rank) where fronts is a list of lists of row indices
    (fronts[0] is the non-dominated set) and rank[i] is the front of row i.
    """
    n = len(M)
    dominated = [[] for _ in range(n)]
    n_dominating = np.zeros(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(M[p], M[q]):
                dominated[p].append(q)
            elif _dominates(M[q], M[p]):
                n_dominating[p] += 1
        if n_dominating[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominated[p]:
                n_dominating[q] -= 1
                if n_dominating[q] == 0:
                    rank[q] = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts, rank


def crowding_distance(M_front):
    """NSGA-II crowding distance for the rows of one front (maximization)."""
    m = len(M_front)
    if m == 0:
        return np.array([])
    dist = np.zeros(m)
    for k in range(M_front.shape[1]):
        order = np.argsort(M_front[:, k])
        dist[order[0]] = dist[order[-1]] = np.inf
        vmin, vmax = M_front[order[0], k], M_front[order[-1], k]
        span = vmax - vmin
        if span == 0:
            continue
        for i in range(1, m - 1):
            dist[order[i]] += (M_front[order[i + 1], k] - M_front[order[i - 1], k]) / span
    return dist


# ============================================================================
# Generic problem + engine
# ============================================================================

class Problem:
    """Base class. Subclasses declare genes + objectives and implement evaluate()."""

    categoricals = {}                 # name -> list of options
    numeric_bounds = {}               # name -> (lo, hi, is_int)
    objectives = []                   # list of (metric_name, 'max'|'min')

    def genes(self):
        return list(self.categoricals) + list(self.numeric_bounds)

    def random_genome(self, rng):
        g = {}
        for k, opts in self.categoricals.items():
            g[k] = opts[int(rng.integers(len(opts)))]
        for k, (lo, hi, is_int) in self.numeric_bounds.items():
            val = rng.uniform(lo, hi)
            g[k] = int(round(val)) if is_int else float(val)
        return g

    def mutate(self, g, rng, rate):
        g = dict(g)
        for k, opts in self.categoricals.items():
            if rng.random() < rate:
                g[k] = opts[int(rng.integers(len(opts)))]
        for k, (lo, hi, is_int) in self.numeric_bounds.items():
            if rng.random() < rate:
                val = g[k] + (hi - lo) * 0.2 * rng.standard_normal()
                val = min(max(val, lo), hi)
                g[k] = int(round(val)) if is_int else float(val)
        return g

    def crossover(self, g1, g2, rng):
        return {k: (g1[k] if rng.random() < 0.5 else g2[k]) for k in self.genes()}

    def evaluate(self, genome):
        raise NotImplementedError


class EvolutionEngine:
    """NSGA-II-lite: Pareto-rank + crowding selection, elitist (mu+lambda) survival."""

    def __init__(self, problem, pop_size=16, generations=8, crossover_rate=0.9,
                 mutation_rate=0.2, seed=0):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.n_evaluations = 0

    def _objective_matrix(self, evals):
        dirs = self.problem.objectives
        return np.array([[(e[name] if d == "max" else -e[name]) for name, d in dirs]
                         for e in evals], dtype=float)

    def _evaluate(self, genome):
        self.n_evaluations += 1
        return self.problem.evaluate(genome)

    def _tournament(self, rng, rank, cd):
        a, b = int(rng.integers(len(rank))), int(rng.integers(len(rank)))
        if rank[a] != rank[b]:
            return a if rank[a] < rank[b] else b
        return a if cd[a] >= cd[b] else b

    def run(self, verbose=True):
        rng = np.random.default_rng(self.seed)
        pop = [self.problem.random_genome(rng) for _ in range(self.pop_size)]
        evals = [self._evaluate(g) for g in pop]
        history = []

        for gen in range(self.generations):
            # Rank current population for parent selection.
            M = self._objective_matrix(evals)
            fronts, rank = non_dominated_sort(M)
            cd = np.zeros(len(pop))
            for fr in fronts:
                cd[fr] = crowding_distance(M[fr])

            # Produce offspring.
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament(rng, rank, cd)
                p2 = self._tournament(rng, rank, cd)
                if rng.random() < self.crossover_rate:
                    child = self.problem.crossover(pop[p1], pop[p2], rng)
                else:
                    child = dict(pop[p1])
                child = self.problem.mutate(child, rng, self.mutation_rate)
                offspring.append(child)
            off_evals = [self._evaluate(g) for g in offspring]

            # Elitist (mu+lambda) survival: combine, sort into fronts, fill.
            allpop = pop + offspring
            allev = evals + off_evals
            M = self._objective_matrix(allev)
            fronts, _ = non_dominated_sort(M)
            newpop, newev = [], []
            for fr in fronts:
                if len(newpop) + len(fr) <= self.pop_size:
                    for i in fr:
                        newpop.append(allpop[i])
                        newev.append(allev[i])
                else:
                    cd = crowding_distance(M[fr])
                    order = sorted(range(len(fr)), key=lambda j: -cd[j])
                    for j in order[:self.pop_size - len(newpop)]:
                        newpop.append(allpop[fr[j]])
                        newev.append(allev[fr[j]])
                    break
            pop, evals = newpop, newev

            front0 = non_dominated_sort(self._objective_matrix(evals))[0][0]
            best = {name: (max if d == "max" else min)(evals[i][name] for i in front0)
                    for name, d in self.problem.objectives}
            history.append({"generation": gen, "front_size": len(front0), "best": best})
            if verbose:
                bstr = ", ".join(f"{k}={v:.3f}" for k, v in best.items())
                print(f"  gen {gen:2d}: front={len(front0):2d}  best[{bstr}]")

        front0 = non_dominated_sort(self._objective_matrix(evals))[0][0]
        front0 = sorted(front0, key=lambda i: evals[i][self.problem.objectives[0][0]],
                        reverse=True)
        return {
            "pareto_front": [{"genome": pop[i], "objectives": evals[i]} for i in front0],
            "population": [{"genome": g, "objectives": e} for g, e in zip(pop, evals)],
            "history": history,
            "n_evaluations": self.n_evaluations,
        }


# ============================================================================
# Problem 1: evolve a spiking reservoir (accuracy vs energy)
# ============================================================================

class NetworkProblem(Problem):
    name = "network"
    objectives = [("accuracy", "max"), ("energy_nj", "min")]
    categoricals = {"topology": list(TOPOLOGIES)}
    numeric_bounds = {
        "reservoir_N": (120, 350, True),
        "conn_prob": (0.05, 0.30, False),
        "rec_scale": (2.0, 40.0, False),
        "input_scale": (4.0, 16.0, False),
        "exc_frac": (0.5, 0.9, False),
        "noise_scale": (1.0, 6.0, False),
    }

    def __init__(self, cfg):
        self.cfg = cfg
        self.eval_seed = cfg["eval_seed"]
        self.n_classes = cfg["eval_n_classes"]
        X, y, meta = generate_temporal_dataset(
            n_per_class=cfg["eval_n_per_class"], n_classes=cfg["eval_n_classes"],
            N_in=cfg["eval_N_in"], T_task=cfg["eval_T"], high_rate_hz=120.0,
            seed=cfg["eval_seed"])
        self.Xtr, self.ytr, self.Xte, self.yte = train_test_split(
            X, y, test_frac=0.3, seed=cfg["eval_seed"])
        self.meta = meta
        self.input_spikes_per_sample = float(self.Xte.sum()) / len(self.Xte)

    def evaluate(self, g):
        res = IzhikevichReservoir(
            N=int(g["reservoir_N"]), N_in=self.meta["N_in"], recurrent=True,
            topology=g["topology"], conn_prob=g["conn_prob"],
            input_scale=g["input_scale"], rec_scale=g["rec_scale"],
            noise_scale=g["noise_scale"], exc_frac=g["exc_frac"], seed=self.eval_seed)
        Phi_tr = res.transform(self.Xtr)
        Phi_te, rate = res.transform(self.Xte, return_rates_hz=True)
        readout = RidgeReadout(alpha=10.0).fit(Phi_tr, self.ytr)
        acc = float(np.mean(readout.predict(Phi_te) == self.yte))
        energy = _snn_energy_nj(res, rate, self.meta["T_task"],
                                self.input_spikes_per_sample, self.n_classes)
        return {"accuracy": acc, "energy_nj": energy}


# ============================================================================
# Problem 2: evolve a synthetic activity pattern (two descriptive metrics)
# ============================================================================

PATTERN_FUNCS = {
    "traveling_wave": generate_traveling_wave,
    "synchronized_bursts": generate_synchronized_bursts,
    "spiral_wave": generate_spiral_wave,
    "game_of_life": generate_game_of_life,
    "random_activity": generate_random_activity,
    "localized_clusters": generate_localized_clusters,
    "ripple_pattern": generate_ripple_pattern,
    "checkerboard": generate_checkerboard_pattern,
}

_INT_PARAMS = {"spiral_arms", "num_clusters", "frequency", "square_size",
               "flip_interval", "wave_width", "cluster_size"}


class PatternProblem(Problem):
    name = "pattern"
    categoricals = {"pattern_name": list(PATTERN_FUNCS)}
    # Union of every generator's parameters (extras are ignored per pattern).
    numeric_bounds = {
        "wave_speed": (0.2, 3.0, False),
        "wave_width": (5, 80, True),
        "frequency": (1, 30, True),
        "participation": (0.1, 1.0, False),
        "rotation_speed": (0.01, 0.20, False),
        "spiral_arms": (1, 5, True),
        "initial_fill": (0.05, 0.50, False),
        "firing_probability": (0.001, 0.10, False),
        "num_clusters": (1, 8, True),
        "cluster_size": (5, 60, True),
        "activity_rate": (0.1, 0.9, False),
        "ripple_speed": (0.1, 2.0, False),
        "square_size": (2, 12, True),
        "flip_interval": (20, 200, True),
    }

    def __init__(self, cfg):
        self.cfg = cfg
        self.objectives = cfg.get("pattern_objectives",
                                  [("synchrony_index", "max"), ("wave_score", "max")])
        self.Ne, self.Ni, self.T = cfg["pattern_Ne"], cfg["pattern_Ni"], cfg["pattern_T"]

    def evaluate(self, g):
        fn = PATTERN_FUNCS[g["pattern_name"]]
        sig = inspect.signature(fn)
        params = {}
        for k in self.numeric_bounds:
            if k in sig.parameters:
                params[k] = int(round(g[k])) if k in _INT_PARAMS else g[k]
        firings = fn(self.Ne, self.Ni, self.T, **params)
        if len(firings) == 0:
            return {name: 0.0 for name, _ in self.objectives}
        m = calculate_all_metrics(firings, self.Ne, self.Ni, self.T)
        return {name: float(m.get(name, 0.0)) for name, _ in self.objectives}


# ============================================================================
# Orchestration
# ============================================================================

EVO_DEFAULTS = {
    "target": "network",          # 'network' or 'pattern'
    "pop_size": 16,
    "generations": 8,
    "crossover_rate": 0.9,
    "mutation_rate": 0.25,
    "seed": 0,
    # network-target evaluation (kept small so evolution is fast)
    "eval_n_per_class": 60,
    "eval_n_classes": 4,
    "eval_N_in": 24,
    "eval_T": 120,
    "eval_seed": 0,
    # pattern-target evaluation
    "pattern_objectives": [("synchrony_index", "max"), ("wave_score", "max")],
    "pattern_Ne": 100,
    "pattern_Ni": 50,
    "pattern_T": 1000,
    # output
    "output_dir": "evolution_results",
    "make_plot": True,
    "verbose": True,
}


def _make_problem(cfg):
    if cfg["target"] == "network":
        return NetworkProblem(cfg)
    if cfg["target"] == "pattern":
        return PatternProblem(cfg)
    raise ValueError(f"Unknown evolution target '{cfg['target']}' (use 'network' or 'pattern').")


def run_evolution(config=None):
    """Run the multi-objective evolution and return a result dict (also saved)."""
    cfg = {**EVO_DEFAULTS, **(config or {})}
    v = cfg["verbose"]
    problem = _make_problem(cfg)
    (o0, d0), (o1, d1) = problem.objectives

    if v:
        print("\n" + "=" * 70)
        print(f"EVOLUTION  -  target='{problem.name}', multi-objective (Pareto)")
        print("=" * 70)
        print(f"Objectives: maximize/minimize {o0} ({d0}) vs {o1} ({d1})")
        print(f"Population {cfg['pop_size']}, generations {cfg['generations']}, seed {cfg['seed']}")

    engine = EvolutionEngine(
        problem, pop_size=cfg["pop_size"], generations=cfg["generations"],
        crossover_rate=cfg["crossover_rate"], mutation_rate=cfg["mutation_rate"],
        seed=cfg["seed"])
    result = engine.run(verbose=v)

    front = result["pareto_front"]
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target": problem.name,
        "objectives": problem.objectives,
        "config": {k: cfg[k] for k in cfg if k != "verbose"},
        "n_evaluations": result["n_evaluations"],
        "pareto_front": front,
        "population": result["population"],
        "history": result["history"],
    }

    if v:
        print("\n" + "=" * 70)
        print(f"PARETO FRONT  ({len(front)} non-dominated solutions, "
              f"{result['n_evaluations']} evaluations)")
        print("=" * 70)
        for p in front:
            obj = "  ".join(f"{k}={p['objectives'][k]:.3f}" for k, _ in problem.objectives)
            extra = {k: (round(val, 3) if isinstance(val, float) else val)
                     for k, val in p["genome"].items()}
            print(f"  {obj}")
            print(f"      genome: {extra}")
        print("=" * 70)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(cfg["output_dir"], f"evolution_{problem.name}_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if v:
        print(f"\nResults saved: {json_path}")

    if cfg["make_plot"]:
        plot_path = os.path.join(cfg["output_dir"], f"evolution_{problem.name}_{stamp}.png")
        try:
            _plot_pareto(result, problem, plot_path)
            results["plot_path"] = plot_path
            if v:
                print(f"Pareto chart:  {plot_path}")
        except Exception as e:
            print(f"(plot skipped: {e})")

    return results


def _plot_pareto(result, problem, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (o0, d0), (o1, d1) = problem.objectives
    pop = result["population"]
    front = sorted(result["pareto_front"], key=lambda p: p["objectives"][o1])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter([p["objectives"][o1] for p in pop], [p["objectives"][o0] for p in pop],
               c="#c7c7c7", s=40, label="final population", zorder=2)
    fx = [p["objectives"][o1] for p in front]
    fy = [p["objectives"][o0] for p in front]
    ax.plot(fx, fy, c="#2a9d8f", lw=1.5, zorder=3)
    ax.scatter(fx, fy, c="#2a9d8f", s=90, label="Pareto front", zorder=4)
    if o1 == "energy_nj":
        ax.set_xscale("log")
    ax.set_xlabel(f"{o1}  ({d1})")
    ax.set_ylabel(f"{o0}  ({d0})")
    ax.set_title(f"Evolved Pareto front - target='{problem.name}'")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
