# run_competition.py
"""
RSNN System - Topology Competition entry point
==============================================
Does network topology matter? This pits three model families against each other
on one shared, rate-matched temporal classification task:

    1. RSNN reservoir           - recurrent Izhikevich network (topology-rich)
    2. Feedforward-SNN reservoir - same neurons, recurrence removed (control)
    3. Traditional NN           - from-scratch NumPy MLP on the raw input

Edit the CONFIG block below, then run:

    python run_competition.py

Outputs (test accuracy, timing, model sizes, verdict, chart + JSON) are written
to ./competition_results/.
"""

import sys

# Ensure emoji/banner output doesn't crash on Windows consoles using a legacy
# code page (cp1252) when stdout is piped or redirected to a file.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from competition import run_competition

# ============================================================================
# CONFIG - edit these to control the competition
# ============================================================================
CONFIG = {
    # --- Task ---
    "n_per_class": 150,     # samples per class
    "n_groups": 3,          # input groups = number of temporal epochs
    "n_classes": 4,         # distinct group orderings (<= n_groups!)
    "N_in": 30,             # input channels (rounded to a multiple of n_groups)
    "T_task": 180,          # trial length (timesteps / ms)
    "high_rate_hz": 120.0,  # active-group firing rate
    "low_rate_hz": 10.0,    # baseline firing rate
    "test_frac": 0.25,      # held-out fraction

    # --- Reservoir ---
    "reservoir_N": 300,     # reservoir neurons (80% exc / 20% inh)
    # One spiking reservoir is run per topology below (the heart of the study):
    "topologies": ["random", "ring", "smallworld", "modular", "scalefree"],
    "include_feedforward": True,  # add a recurrence-removed control
    "conn_prob": 0.12,      # connection density (matched across topologies)
    "input_conn_prob": 0.3, # input -> reservoir connection probability
    "input_scale": 10.0,    # input current strength
    "rec_scale": 14.0,      # recurrent gain (used only if match_firing_hz is None)
    "match_firing_hz": 30.0,  # calibrate each topology to this rate (None to disable)
    "calib_samples": 40,    # samples used to calibrate the firing rate
    "noise_scale": 3.0,     # background current noise
    "ridge_alpha": 10.0,    # readout L2 regularization

    # --- Traditional NN (NumPy MLP) ---
    "mlp_hidden": 128,
    "mlp_lr": 1e-3,
    "mlp_epochs": 120,
    "mlp_batch_size": 64,

    # --- Experiment ---
    "seeds": [0, 1, 2],     # repeated runs for error bars
    "output_dir": "competition_results",
    "make_plot": True,
    "verbose": True,
}
# ============================================================================

if __name__ == "__main__":
    run_competition(CONFIG)
