# run_evolution.py
"""
RSNN System - Evolutionary exploration entry point
==================================================
A shared multi-objective (NSGA-II / Pareto) evolutionary engine that can evolve
either of two targets:

    target = 'network'  -> evolve a spiking reservoir (topology + hyperparameters)
                           to trade off task ACCURACY (max) vs ENERGY (min).
    target = 'pattern'  -> evolve a synthetic activity pattern (generator + params)
                           to trade off two descriptive metrics (default:
                           synchrony_index vs wave_score).

Edit the CONFIG block below, then run:

    python run_evolution.py

Outputs (the Pareto front, full population, history, JSON + a Pareto chart) land
in ./evolution_results/.
"""

import sys

# Keep emoji/banner output safe when piped/redirected on Windows (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from competition.evolution import run_evolution

# ============================================================================
# CONFIG - edit these to control the evolution
# ============================================================================
CONFIG = {
    # --- What to evolve ---
    "target": "network",        # 'network' (accuracy vs energy) or 'pattern' (metric vs metric)

    # --- Evolutionary engine ---
    "pop_size": 16,
    "generations": 8,
    "crossover_rate": 0.9,
    "mutation_rate": 0.25,
    "seed": 0,

    # --- Network-target evaluation (kept small so evolution is quick) ---
    "eval_n_per_class": 60,
    "eval_n_classes": 4,
    "eval_N_in": 24,
    "eval_T": 120,
    "eval_seed": 0,

    # --- Pattern-target evaluation ---
    # Two objectives to trade off; each is (metric_name, 'max'|'min') from metrics.py.
    "pattern_objectives": [("synchrony_index", "max"), ("wave_score", "max")],
    "pattern_Ne": 100,
    "pattern_Ni": 50,
    "pattern_T": 1000,

    # --- Output ---
    "output_dir": "evolution_results",
    "make_plot": True,
    "verbose": True,
}
# ============================================================================

if __name__ == "__main__":
    run_evolution(CONFIG)
