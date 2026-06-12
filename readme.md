# RSNN System — Spiking Neural Network Simulation with Autonomous Pattern Exploration

A simulation framework for spiking neural networks (SNNs) using the **Izhikevich neuron
model**, featuring autonomous pattern exploration, a comprehensive metrics suite, and
interactive 2D/3D raster visualization.

> Status: working simulator with autonomous agent, metrics, and visualization. CPU-only
> (pure NumPy/SciPy — no PyTorch, no GPU).

---

## Features

### Autonomous Agent System
- **4 exploration strategies**: `random`, `sequential`, `optimize` (drive a target metric), `diversity`
- **Automatic pattern discovery** across the pattern space
- **Experiment tracking**: every run saved with metrics + metadata (`experiment_tracker.py`)

### Pattern Library (`patterns.py`)
- **8 pattern types**: traveling wave, synchronized bursts, spiral wave, game of life,
  ripple, localized clusters, checkerboard, random activity
- Extensible: register a new generator and it shows up everywhere

### Metrics (`metrics.py`)
- **Firing rate**: mean rate, coefficient of variation (CV)
- **Synchrony**: synchrony index, burst frequency, participation ratio, ISI CV
- **Spatial**: spatial correlation, wave score, activity sparsity
- Export to JSON

### Performance
- `scipy.sparse` connectivity, pre-allocated arrays, `tqdm` progress bars, plot downsampling

---

## Installation

```powershell
# from the project root
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt     # numpy, scipy, matplotlib, plotly, tqdm, pandas
```

---

## Usage

Everything is controlled from the **configuration block at the top of `main.py`** — edit the
variables there, then run:

```powershell
python main.py
```

Pick what runs with `RUN_MODE`:

| `RUN_MODE`     | What it does |
|----------------|--------------|
| `simulation`   | Full Izhikevich simulation with sparse random connectivity |
| `pattern`      | Generate one named pattern (`PATTERN_NAME`) from the library |
| `autonomous`   | Run an exploration agent (`AGENT_STRATEGY`) over `NUM_EXPERIMENTS` runs |
| `compare_all`  | Run every pattern and rank them by metric (also a handy smoke test) |

Other key knobs: `Ne`/`Ni` (neuron counts), `T` (sim ms), `connection_prob`,
`EXCITATORY_TYPE`/`INHIBITORY_TYPE`, the viz flags (`GENERATE_2D_PLOT`, `GENERATE_3D_PLOT`,
`USE_PLOTLY_3D`), and `RANDOM_SEED` (set an int for reproducibility).

---

## Topology Competition — does network structure matter?

A reservoir-computing benchmark (`run_competition.py` + the `competition/` package) that pits
three model families against each other on one shared **rate-matched temporal classification**
task. Each class activates the same input groups for the same total time, differing only in
their *order*, so mean firing rate carries no class information — separating the classes
requires using *when* things happened (temporal processing).

Contestants:

1. **A suite of spiking reservoirs**, one per network **topology** — `random` (Erdős–Rényi),
   `ring` (local lattice), `smallworld` (Watts–Strogatz), `modular` (communities), `scalefree`
   (hubs) — all recurrent Izhikevich networks
2. **Feedforward-SNN reservoir** — same neurons with recurrence removed (the "no topology" control)
3. **Traditional NN** — a from-scratch NumPy MLP on the raw input (no PyTorch, CPU-only)

The spiking reservoirs share an identical ridge readout and, for a given seed, identical neurons
and input projection — **only the recurrent wiring differs**. Topologies are matched on
connection **density** and (via per-topology gain calibration) on mean **firing rate**, so any
accuracy difference reflects *structure*, not activity level.

```powershell
python run_competition.py          # edit the CONFIG block at the top to change anything
python tests/test_competition.py   # fast smoke tests (or: pytest)
```

Outputs land in `competition_results/`: a printed verdict, a results JSON, an accuracy bar
chart, and an accuracy-vs-energy chart. Every contestant is scored on three axes — **accuracy**,
**inference latency** (ms/sample), and **estimated energy** per inference. Energy uses a
hardware-agnostic proxy: synaptic operations for the spiking nets (event-driven: energy ∝
spikes × fan-out) and MACs for the dense MLP, costed with the Horowitz 2014 (45 nm) constants.
Typical finding: the MLP is the most accurate but the spiking reservoirs are ~100×+ leaner on
energy, and among them small-world gives the best accuracy-per-energy.

**Result (3 seeds, N=300, matched 30 Hz, 4 classes):** topology matters **decisively**. At
identical density and firing rate, rewiring alone spans a **0.32** accuracy gap:

| Contestant | Test accuracy |
|---|---|
| Traditional NN (MLP, raw input) | 1.00 ± 0.00 |
| RSNN — ring | 0.94 ± 0.01 |
| RSNN — small-world | 0.94 ± 0.03 |
| RSNN — scale-free | 0.80 ± 0.12 |
| RSNN — random | 0.63 ± 0.06 |
| RSNN — modular | 0.62 ± 0.06 |
| Feedforward-SNN | 0.61 ± 0.04 |
| chance | 0.25 |

The best topology beats the worst on every seed (paired *t* ≈ 6.7). Structured local/small-world
wiring lets the reservoir integrate the spatiotemporal signal; a fragmented modular network
cannot, and random wiring is mediocre. Note the two framings of "topology": **structure vs
structure** is large (ring − modular ≈ +0.32), while **recurrence vs none** (random −
feedforward) is small (≈ +0.02) — because Izhikevich neurons already carry *intrinsic* memory,
so merely *having* recurrence matters far less than *how it is wired*. Knobs live in the `CONFIG`
block: `topologies`, `match_firing_hz`, `reservoir_N`, `n_groups`/`n_classes`.

---

## Evolutionary exploration (optional)

`run_competition.py`'s static suite asks "which of these hand-picked topologies wins?".
`run_evolution.py` asks the open-ended version: a shared **multi-objective evolutionary engine**
(NSGA-II — Pareto-rank + crowding-distance selection, elitist survival) that searches for a whole
*Pareto front* over two competing objectives. One engine, two targets:

- **`target='network'`** — evolve a spiking reservoir's **topology + hyperparameters** (topology,
  size, density, recurrent gain, input scale, E/I ratio, noise) to trade off task **accuracy** (↑)
  against estimated **energy** (↓).
- **`target='pattern'`** — evolve a synthetic generator (which generator + its parameters) to
  trade off two descriptive metrics from `metrics.py` (default `synchrony_index` vs `wave_score`),
  extending the project's original autonomous pattern exploration.

```powershell
python run_evolution.py            # edit CONFIG: target, pop_size, generations, objectives
python tests/test_evolution.py     # fast smoke tests (or: pytest)
```

Outputs (Pareto front, full population, generation history, JSON + a Pareto-front chart) land in
`evolution_results/`.

**Example (network target, accuracy vs energy):** evolution returns a clean frontier from
lean-and-cheap (~0.32 accuracy, ~7 nJ) up to accurate-and-costly (~0.93 accuracy, ~82 nJ), with a
knee around **0.90 accuracy at ~19 nJ**. Tellingly, energy-aware evolution favors compact
**scale-free** reservoirs, whereas the matched-firing-rate competition favored **small-world** —
different objective, different winner, which is the whole point of having both tools.

---

## Project Layout

```
RSNN System/
├── main.py                # Control center: config block + pipeline (entry point)
├── patterns.py            # PatternLibrary + the 8 pattern generators
├── autonomous_agent.py    # Exploration strategies + create_agent()
├── experiment_tracker.py  # Run persistence + metadata
├── metrics.py             # calculate_all_metrics() + summary printing
├── plot_3d_raster.py      # 2D (Matplotlib) + 3D (Plotly/PNG) raster plots
├── neuron_models.py       # Shared Izhikevich [a,b,c,d] presets (used by main + competition)
├── run_competition.py     # Topology-competition entry point (config block + run)
├── run_evolution.py       # Evolutionary-exploration entry point (config block + run)
├── competition/           # task · reservoir · connectivity · classifiers · compete · evolution
├── tests/                 # Smoke tests (test_competition.py, test_evolution.py)
├── requirements.txt       # numpy, scipy, matplotlib, plotly, tqdm, pandas
├── raster_plots/          # Output: 2D Plots/ · 3D Plots/ · Metrics/ · Firing Data/
├── competition_results/   # Output: competition JSON + accuracy / energy charts
├── evolution_results/     # Output: evolution JSON + Pareto-front charts
└── experiments/           # Saved autonomous-exploration runs
```

---

## Network Architecture

- **Neuron model**: Izhikevich `[a, b, c, d]` presets — RS (Regular Spiking), FS (Fast Spiking),
  LTS (Low-Threshold Spiking), IB (Intrinsically Bursting), CH (Chattering)
- **Default populations**: Excitatory = RS, Inhibitory = LTS
- **Connectivity**: sparse random (default 10%)
- **Synaptic weights**: excitatory +0.5, inhibitory −1.0

---

## Metrics Reference

| Group | Metric | Meaning |
|-------|--------|---------|
| Firing rate | Mean rate | Average spikes per neuron per second |
| Firing rate | CV | Variability in firing rates |
| Synchrony | Synchrony index | Variance-to-mean ratio of population activity |
| Synchrony | Burst frequency | Rate of high-activity events |
| Synchrony | Participation ratio | Fraction of active neurons |
| Synchrony | ISI CV | Regularity of inter-spike intervals |
| Spatial | Spatial correlation | Correlation between adjacent neurons |
| Spatial | Wave score | Strength of traveling-wave structure |
| Spatial | Activity sparsity | Fraction of active spatiotemporal bins |
