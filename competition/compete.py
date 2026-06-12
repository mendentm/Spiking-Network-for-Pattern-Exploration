# competition/compete.py
"""
Competition orchestration.
==========================

Runs the contestants on the shared rate-matched temporal task across several
seeds, then reports test accuracy (mean +/- std), training time, model size, and
a verdict on whether network topology matters.

Contestants
-----------
- One spiking reservoir per topology in `topologies` (recurrent Izhikevich net,
  same neurons and input projection for a given seed - only the recurrent wiring
  differs). Read out by an identical ridge classifier.
- A feedforward-SNN reservoir (recurrence removed) - the "no network topology"
  control.
- A traditional NumPy MLP trained on the raw flattened input - the conventional,
  non-spiking reference.

Fair comparison of topologies
-----------------------------
Topologies are matched on (a) connection density and (b) - when
`match_firing_hz` is set - mean firing rate, by calibrating each reservoir's
recurrent gain to a common operating point. So accuracy differences reflect
*structure*, not how active the network happens to be.

Success criterion (measurable): topology matters iff the best topology beats the
worst by more than the seed-to-seed spread (paired across seeds).
"""

import os
import json
import time
from datetime import datetime

import numpy as np

from .task import (generate_temporal_dataset, train_test_split,
                   dataset_rate_summary)
from .reservoir import IzhikevichReservoir
from .classifiers import RidgeReadout, NumpyMLP
from .connectivity import TOPOLOGIES


DEFAULTS = {
    # Task
    "n_per_class": 150,
    "n_groups": 3,
    "n_classes": 4,
    "N_in": 30,
    "T_task": 180,
    "high_rate_hz": 120.0,
    "low_rate_hz": 10.0,
    "test_frac": 0.25,
    # Reservoir
    "reservoir_N": 300,
    "topologies": list(TOPOLOGIES),       # random, ring, smallworld, modular, scalefree
    "include_feedforward": True,          # feedforward-SNN control
    "conn_prob": 0.12,
    "input_conn_prob": 0.3,
    "input_scale": 10.0,
    "rec_scale": 14.0,                    # used if match_firing_hz is None
    "match_firing_hz": 30.0,             # calibrate each topology to this rate (None to disable)
    "calib_samples": 40,
    "noise_scale": 3.0,
    "ridge_alpha": 10.0,
    # Traditional NN
    "mlp_hidden": 128,
    "mlp_lr": 1e-3,
    "mlp_epochs": 120,
    "mlp_batch_size": 64,
    # Experiment
    "seeds": [0, 1, 2],
    "output_dir": "competition_results",
    "make_plot": True,
    "verbose": True,
}

FFSNN = "Feedforward-SNN"
MLP = "Traditional-NN (MLP)"

# Energy proxy constants (Horowitz, ISSCC 2014, 45 nm). Rough + hardware-agnostic:
# SNNs are event-driven, so dynamic cost is counted as the synaptic operations
# (accumulates) triggered by spikes; a dense ANN pays a multiply-accumulate per
# weight. This estimates relative energy, not a measurement on real silicon.
E_AC_PJ = 0.9       # one accumulate           (SNN synaptic op)
E_MAC_PJ = 4.6      # one multiply-accumulate  (ANN op)


def _rsnn_name(topology):
    return f"RSNN ({topology})"


def _snn_energy_nj(res, rate_hz, T, input_spikes_per_sample, n_classes):
    """
    Estimated dynamic energy per inference (nJ) for a spiking reservoir, via a
    SynOps proxy: every spike triggers one accumulate at each downstream synapse.
    Counts recurrent + input synaptic ops plus the linear readout's MACs; excludes
    neuron-state updates and memory traffic (standard simplification).
    """
    spikes_per_sample = rate_hz * res.N * T / 1000.0          # mean Hz -> spikes/trial
    rec_fanout = np.count_nonzero(res.W_rec) / res.N
    in_fanout = np.count_nonzero(res.W_in) / res.N_in
    synops = spikes_per_sample * rec_fanout + input_spikes_per_sample * in_fanout
    readout_macs = res.N * n_classes
    return (synops * E_AC_PJ + readout_macs * E_MAC_PJ) / 1000.0   # pJ -> nJ


def _mlp_energy_nj(input_dim, hidden, n_classes):
    """Estimated energy per inference (nJ) for the dense MLP: one MAC per weight."""
    macs = input_dim * hidden + hidden * n_classes
    return macs * E_MAC_PJ / 1000.0


def _agg(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "runs": arr.tolist()}


def _paired(a, b):
    """Paired difference stats between two equal-length per-seed accuracy lists."""
    gaps = np.asarray(a) - np.asarray(b)
    n = len(gaps)
    mean = float(gaps.mean())
    std = float(gaps.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if (n > 1 and std > 0) else 0.0
    t = float(mean / se) if se > 0 else (float("inf") if (mean > 0 and n > 1) else 0.0)
    return {"mean": mean, "std": std, "t": t, "wins": int((gaps > 0).sum()),
            "n": n, "per_seed": gaps.tolist()}


def run_competition(config=None):
    """Run the full competition and return a results dict (also saved to disk)."""
    cfg = {**DEFAULTS, **(config or {})}
    v = cfg["verbose"]
    topologies = list(cfg["topologies"])

    contestants = [_rsnn_name(t) for t in topologies]
    if cfg["include_feedforward"]:
        contestants.append(FFSNN)
    contestants.append(MLP)

    if v:
        print("\n" + "=" * 70)
        print("TOPOLOGY COMPETITION  -  does network structure matter?")
        print("=" * 70)
        print(f"Task: rate-matched temporal sequence classification "
              f"({cfg['n_classes']} classes, chance = {1.0/cfg['n_classes']:.3f})")
        match = (f"matched to {cfg['match_firing_hz']} Hz"
                 if cfg["match_firing_hz"] else f"rec_scale={cfg['rec_scale']}")
        print(f"Reservoir: N={cfg['reservoir_N']}, topologies={topologies} ({match})")
        print(f"Seeds: {cfg['seeds']}")

    acc = {name: [] for name in contestants}
    train_time = {name: [] for name in contestants}
    inference_ms = {name: [] for name in contestants}
    energy_nj = {name: [] for name in contestants}
    n_params = {name: None for name in contestants}
    res_rates = {name: [] for name in contestants if name != MLP}
    rate_match = []

    for seed in cfg["seeds"]:
        if v:
            print(f"\n--- seed {seed} ---")

        X, y, meta = generate_temporal_dataset(
            n_per_class=cfg["n_per_class"], n_groups=cfg["n_groups"],
            n_classes=cfg["n_classes"], N_in=cfg["N_in"], T_task=cfg["T_task"],
            high_rate_hz=cfg["high_rate_hz"], low_rate_hz=cfg["low_rate_hz"],
            seed=seed)
        Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=cfg["test_frac"], seed=seed)
        rate_match.append(dataset_rate_summary(X, y))

        def build(recurrent, topology):
            return IzhikevichReservoir(
                N=cfg["reservoir_N"], N_in=meta["N_in"], recurrent=recurrent,
                topology=topology, conn_prob=cfg["conn_prob"],
                input_conn_prob=cfg["input_conn_prob"], input_scale=cfg["input_scale"],
                rec_scale=cfg["rec_scale"], noise_scale=cfg["noise_scale"], seed=seed)

        # ---- One reservoir per topology + the feedforward control ----
        spiking = [(_rsnn_name(t), True, t) for t in topologies]
        if cfg["include_feedforward"]:
            spiking.append((FFSNN, False, "random"))

        for name, recurrent, topology in spiking:
            res = build(recurrent, topology)
            if recurrent and cfg["match_firing_hz"]:
                res.calibrate_rec_scale(Xtr[:cfg["calib_samples"]], cfg["match_firing_hz"])

            # Training cost: build train-set features + fit the linear readout.
            t0 = time.perf_counter()
            Phi_tr = res.transform(Xtr)
            readout = RidgeReadout(alpha=cfg["ridge_alpha"]).fit(Phi_tr, ytr)
            train_time[name].append(time.perf_counter() - t0)

            # Inference cost: run the reservoir on the test set + read out.
            t0 = time.perf_counter()
            Phi_te, rate_hz = res.transform(Xte, return_rates_hz=True)
            preds = readout.predict(Phi_te)
            inf_t = time.perf_counter() - t0

            a = float(np.mean(preds == yte))
            acc[name].append(a)
            res_rates[name].append(rate_hz)
            inference_ms[name].append(inf_t / len(yte) * 1000.0)
            energy_nj[name].append(_snn_energy_nj(
                res, rate_hz, meta["T_task"], float(Xte.sum()) / len(Xte),
                cfg["n_classes"]))
            n_params[name] = res.n_params() + readout.n_params()
            if v:
                print(f"  {name:22s} acc={a:.3f}  ({rate_hz:4.1f} Hz, "
                      f"{inference_ms[name][-1]:.2f} ms/sample, "
                      f"{energy_nj[name][-1]:.0f} nJ)")

        # ---- Traditional NN on raw flattened input ----
        Xtr_flat = Xtr.reshape(len(Xtr), -1)
        Xte_flat = Xte.reshape(len(Xte), -1)
        t0 = time.perf_counter()
        mlp = NumpyMLP(hidden=cfg["mlp_hidden"], lr=cfg["mlp_lr"],
                       epochs=cfg["mlp_epochs"], batch_size=cfg["mlp_batch_size"],
                       seed=seed).fit(Xtr_flat, ytr)
        train_time[MLP].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        preds = mlp.predict(Xte_flat)
        inf_t = time.perf_counter() - t0
        a = float(np.mean(preds == yte))
        acc[MLP].append(a)
        inference_ms[MLP].append(inf_t / len(yte) * 1000.0)
        energy_nj[MLP].append(_mlp_energy_nj(Xtr_flat.shape[1], cfg["mlp_hidden"],
                                             cfg["n_classes"]))
        n_params[MLP] = mlp.n_params()
        if v:
            print(f"  {MLP:22s} acc={a:.3f}  ({inference_ms[MLP][-1]:.2f} ms/sample, "
                  f"{energy_nj[MLP][-1]:.0f} nJ)")

    chance = 1.0 / cfg["n_classes"]
    summary = {name: _agg(acc[name]) for name in contestants}

    # Topology ranking + best-vs-worst (the headline "does topology matter").
    topo_means = {t: summary[_rsnn_name(t)]["mean"] for t in topologies}
    ranked = sorted(topo_means, key=lambda t: -topo_means[t])
    best_t, worst_t = ranked[0], ranked[-1]
    best_vs_worst = _paired(acc[_rsnn_name(best_t)], acc[_rsnn_name(worst_t)])
    topology_matters = (len(topologies) >= 2 and best_vs_worst["mean"] > 0.05
                        and best_vs_worst["t"] >= 2.0)

    # Secondary comparisons.
    structure_vs_random = None
    if "random" in topologies and best_t != "random":
        structure_vs_random = _paired(acc[_rsnn_name(best_t)], acc[_rsnn_name("random")])
    recurrence_vs_ff = None
    if cfg["include_feedforward"] and "random" in topologies:
        recurrence_vs_ff = _paired(acc[_rsnn_name("random")], acc[FFSNN])

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {k: cfg[k] for k in cfg if k != "verbose"},
        "task_meta": {k: meta[k] for k in ("n_classes", "n_groups", "group_size",
                                           "N_in", "T_task", "epoch_len",
                                           "chance_level", "orders")},
        "chance_level": chance,
        "accuracy": summary,
        "train_time_s": {name: _agg(train_time[name]) for name in train_time},
        "inference_ms_per_sample": {name: _agg(inference_ms[name]) for name in contestants},
        "energy_nj_per_inference": {name: _agg(energy_nj[name]) for name in contestants},
        "energy_model": {"E_AC_pJ": E_AC_PJ, "E_MAC_pJ": E_MAC_PJ,
                         "note": "SynOps (accumulates) for SNN, MACs for ANN; Horowitz "
                                 "2014 45nm; excludes neuron-state updates and memory"},
        "n_params": n_params,
        "reservoir_firing_hz": {name: _agg(res_rates[name]) for name in res_rates},
        "rate_match_per_class": rate_match[0],
        "topology_ranking": [(t, topo_means[t]) for t in ranked],
        "best_topology": best_t,
        "worst_topology": worst_t,
        "best_vs_worst": best_vs_worst,
        "structure_vs_random": structure_vs_random,
        "recurrence_vs_feedforward": recurrence_vs_ff,
        "topology_matters": bool(topology_matters),
    }

    if v:
        _print_verdict(results, chance)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(cfg["output_dir"], f"competition_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if v:
        print(f"\nResults saved: {json_path}")

    if cfg["make_plot"]:
        plot_path = os.path.join(cfg["output_dir"], f"competition_{stamp}.png")
        try:
            _make_plot(summary, chance, plot_path, cfg, topologies)
            results["plot_path"] = plot_path
            if v:
                print(f"Chart saved:   {plot_path}")
        except Exception as e:                # plotting must never sink the run
            print(f"(plot skipped: {e})")

        energy_path = os.path.join(cfg["output_dir"], f"competition_energy_{stamp}.png")
        try:
            _make_energy_plot(summary, results["energy_nj_per_inference"], energy_path, cfg)
            results["energy_plot_path"] = energy_path
            if v:
                print(f"Energy chart:  {energy_path}")
        except Exception as e:
            print(f"(energy plot skipped: {e})")

    return results


def _print_verdict(results, chance):
    acc = results["accuracy"]
    print("\n" + "=" * 70)
    print("RESULTS  (test accuracy, mean +/- std over seeds)")
    print("=" * 70)
    for name in sorted(acc, key=lambda k: -acc[k]["mean"]):
        a = acc[name]
        print(f"  {name:24s} {a['mean']:.3f} +/- {a['std']:.3f}")
    print(f"  {'chance':24s} {chance:.3f}")

    inf = results["inference_ms_per_sample"]
    en = results["energy_nj_per_inference"]
    print("\nEFFICIENCY  (inference latency + estimated energy per sample)")
    print("-" * 70)
    print(f"  {'contestant':24s} {'acc':>6} {'ms/sample':>10} {'energy(nJ)':>11}")
    for name in sorted(acc, key=lambda k: -acc[k]["mean"]):
        print(f"  {name:24s} {acc[name]['mean']:6.3f} {inf[name]['mean']:10.2f} "
              f"{en[name]['mean']:11.0f}")
    spiking = [n for n in en if n != MLP]
    if spiking and MLP in en:
        cheapest = min(spiking, key=lambda n: en[n]["mean"])
        if en[cheapest]["mean"] > 0:
            ratio = en[MLP]["mean"] / en[cheapest]["mean"]
            print(f"  -> MLP uses ~{ratio:.0f}x the energy of the leanest spiking net "
                  f"('{cheapest}')")

    bw = results["best_vs_worst"]
    best, worst = results["best_topology"], results["worst_topology"]
    t_str = "inf" if bw["t"] == float("inf") else f"{bw['t']:.2f}"
    print("\nTOPOLOGY EFFECT  (best vs worst topology, matched density/firing rate)")
    print("-" * 70)
    print(f"  best='{best}'  worst='{worst}'  gap={bw['mean']:+.3f} +/- {bw['std']:.3f}  "
          f"(best wins {bw['wins']}/{bw['n']} seeds, paired t={t_str})")
    if results["structure_vs_random"] is not None:
        s = results["structure_vs_random"]
        print(f"  structure vs random: '{best}' - 'random' = {s['mean']:+.3f}")
    if results["recurrence_vs_feedforward"] is not None:
        r = results["recurrence_vs_feedforward"]
        print(f"  recurrence vs none:  'random' - feedforward = {r['mean']:+.3f}")

    print("\nVERDICT")
    print("-" * 70)
    if results["topology_matters"]:
        print(f"  Topology MATTERS, decisively: at matched density and firing rate,")
        print(f"  rewiring alone moves accuracy by {bw['mean']:.3f} "
              f"('{best}' over '{worst}').")
        print(f"  Same neurons, same inputs, same activity level - only structure differs.")
    else:
        print(f"  No clear topology effect here (best-worst gap {bw['mean']:+.3f}).")
        print(f"  Try more seeds, larger reservoir_N, or a harder task.")
    print("=" * 70)


def _make_plot(summary, chance, path, cfg, topologies):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(summary, key=lambda k: -summary[k]["mean"])
    means = [summary[n]["mean"] for n in names]
    stds = [summary[n]["std"] for n in names]

    def color(n):
        if n == MLP:
            return "#9aa0a6"          # gray = conventional reference
        if n == FFSNN:
            return "#e76f51"          # orange = no-topology control
        return "#2a9d8f"              # teal = spiking reservoir topologies

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(names)), 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=[color(n) for n in names], alpha=0.9)
    ax.axhline(chance, ls="--", c="gray", lw=1.5, label=f"chance ({chance:.2f})")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.12)
    match = (f"matched {cfg['match_firing_hz']:.0f} Hz"
             if cfg["match_firing_hz"] else f"rec_scale {cfg['rec_scale']}")
    ax.set_title("Does topology matter? Rate-matched temporal classification\n"
                 f"(N={cfg['reservoir_N']}, {match}, {len(cfg['seeds'])} seeds)")
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, min(m + s + 0.03, 1.10),
                f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _make_energy_plot(summary_acc, energy, path, cfg):
    """Accuracy vs estimated energy per inference (up-and-left = better)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary_acc)

    def color(n):
        if n == MLP:
            return "#9aa0a6"
        if n == FFSNN:
            return "#e76f51"
        return "#2a9d8f"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for n in names:
        x = max(energy[n]["mean"], 1e-6)
        y = summary_acc[n]["mean"]
        ax.scatter(x, y, s=110, color=color(n), zorder=3)
        ax.annotate(n, (x, y), fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Estimated energy per inference  (nJ, log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.1)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("Accuracy vs energy  (top-left is better)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
