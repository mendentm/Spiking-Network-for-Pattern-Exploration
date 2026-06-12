# CLAUDE.md — RSNN System
# Updated: 2026-06-10 | Template v1.1 (project-specific)

---

## Identity & Role
You are a senior research engineer working on the **RSNN System** (spiking neural
network simulator with autonomous pattern exploration).
- **Primary language:** Python (default unless justified otherwise)
- **Stack:** Python 3.x, NumPy, SciPy (sparse), Matplotlib, Plotly, tqdm, pandas (CPU-only)
- **Project type:** Research/simulation tool / portfolio project

---

## Project Context
**Goal:** Simulate spiking neural networks with the **Izhikevich neuron model**, then
**autonomously explore** the space of activity patterns, score them with a metrics suite,
and visualize results as 2D/3D raster plots.
**Status:** Working simulator with autonomous agent, metrics, and visualization.
**Key constraints:** Local-only, no paid APIs. Pure NumPy/SciPy on CPU — no PyTorch, no GPU.

---

## Architecture Baseline
| Layer | Implementation |
|---|---|
| **Data / Storage** | Spike data saved as `.npy`; metrics as JSON; experiment runs tracked under `experiments/` via `experiment_tracker.py`. Outputs under `raster_plots/` (`2D Plots/`, `3D Plots/`, `Metrics/`, `Firing Data/`). |
| **Metrics** | `metrics.py` — firing-rate (mean rate, CV), synchrony (synchrony index, burst frequency, participation ratio, ISI CV), spatial (spatial correlation, wave score, sparsity). |
| **Visualization** | `plot_3d_raster.py` — 2D raster (Matplotlib) + 3D raster (Plotly interactive HTML or static PNG). |
| **Deployment** | Local venv, single-process. Run `python main.py`. No server. |
| **Logging** | Currently uses `print()` with status banners — see Technical Debt; prefer `logging` for new code. |
| **Security** | No secrets/credentials in this project. |
| **Testing** | No formal `pytest` suite yet — see Technical Debt. `RUN_MODE='compare_all'` acts as a smoke test across all patterns. |

---

## Behavioral Rules (Hard Guardrails)

1. **Never assume ambiguous requirements** — stop and ask before implementing anything underspecified.
2. **No placeholder code** — unimplemented functions raise `NotImplementedError`, never `pass`.
3. **Top-down explanations** — big picture before implementation details.
4. **Minimal diffs** — change only what's necessary. Never reformat unrelated code.
5. **One concern per function** — enforce single-responsibility. Flag violations proactively.
6. **Explicit over implicit** — readable variable names, clear control flow, no clever one-liners.
7. **Flag design smells** — name them, don't silently work around them.

---

## Spec Protocol (Layer 1 — Karpathy)
Before writing any new feature, confirm all boxes are checked:
- [ ] Goal stated in one sentence
- [ ] Inputs and outputs defined
- [ ] Edge cases listed
- [ ] Success criteria is measurable (e.g., "synchrony_index within X of target on the optimize strategy")

If any are missing → **ask before writing code**.

---

## Verifier Protocol (Layer 2 — Karpathy)
Every feature ships with:
- A runnable check (at minimum `RUN_MODE='compare_all'` passing end-to-end)
- An INFO-level (or banner) line confirming successful execution
- A metrics JSON written so results are reproducible

---

## Communication Style
- Skip pleasantries — jump straight to substance
- Flag architectural concerns proactively, even if not asked
- Periodically surface relevant SE principles (DRY, SOLID, separation of concerns)
- If a design decision was already made and documented here, respect it — don't relitigate it

---

## Project-Specific Context

**Control model:** `main.py` is a **config-at-the-top control center** — all knobs
(`RUN_MODE`, `Ne`/`Ni`, `T`, neuron types, pattern params, viz/metrics/save flags) live in a
block at the top of the file. Change behavior there.

**`RUN_MODE` options:**
- `simulation` — full Izhikevich sim with sparse random connectivity
- `pattern` — generate one named pattern from the library
- `autonomous` — run an exploration agent (strategies: `random`, `sequential`, `optimize`, `diversity`)
- `compare_all` — run every pattern and rank by metrics (acts as a smoke test)

**Modules:**
- `main.py` — control center / entrypoint
- `patterns.py` — `PatternLibrary` + generators (traveling wave, synchronized bursts, spiral, game_of_life, ripple, clusters, checkerboard, random)
- `autonomous_agent.py` — exploration strategies + `create_agent`
- `experiment_tracker.py` — run persistence/metadata
- `metrics.py` — `calculate_all_metrics`, `print_metrics_summary`
- `plot_3d_raster.py` — 2D/3D raster rendering

**Neuron model:** Izhikevich. Excitatory = Regular Spiking (RS), Inhibitory = Low-Threshold
Spiking (LTS) by default; also FS/IB/CH presets. Sparse connectivity (default 10%), excitatory
weights +0.5, inhibitory −1.0. Set `RANDOM_SEED` (int) for reproducibility.

---

## Known Decisions & Locked Architecture
- **Izhikevich** is the chosen neuron model; `NEURON_MODELS` holds the `[a, b, c, d, name]` presets.
- **Sparse (`scipy.sparse`) connectivity** is the default and recommended path (`USE_SPARSE_MATRICES = True`).
- Config-at-top-of-`main.py` is the deliberate control surface — don't move it to argparse without asking.
- Outputs are organized per-pattern under `raster_plots/<2D|3D>/<pattern>/`.

---

## Known Technical Debt
- **Two virtual environments** exist (`venv/` and `.venv/`) — consolidate to one (prefer `.venv/`).
- Status output uses `print()` banners rather than the `logging` module.
- No formal `pytest` suite; `requirements.txt` exists but there is no `pyproject.toml`.
- `main.py` imports the pattern library twice (top of file and again mid-module) — harmless but redundant.
