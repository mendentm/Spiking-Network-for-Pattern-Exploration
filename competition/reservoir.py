# competition/reservoir.py
"""
Izhikevich spiking reservoir (the spiking contestants).
=======================================================

A fixed (untrained) recurrent Izhikevich network used as a reservoir: it
transforms an input spike train into a high-dimensional dynamical response, and
a simple linear readout is trained on top (see classifiers.RidgeReadout).

Two modes, identical in every respect except recurrence:
    recurrent=True   -> RSNN reservoir (topology-rich)
    recurrent=False  -> feedforward-SNN reservoir (recurrent weights zeroed)

The neuron parameters and connectivity statistics mirror
`main.setup_network` (RS excitatory + LTS inhibitory, exc weights +0.5*U,
inh weights -1.0*U, sparse mask) so the comparison stays faithful to the
project's own simulator. Shared presets come from `neuron_models.NEURON_MODELS`.

Readout features: the **time-averaged firing rate per neuron** over the whole
trial. This is deliberately memoryless at the readout - so order information
survives to the classifier only if the network's own dynamics carried it across
time. A feedforward reservoir cannot do this (its averaged response is a fixed
function of the rate-matched input); a recurrent one can. That asymmetry is the
experiment.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_models import NEURON_MODELS
from .connectivity import make_adjacency


class IzhikevichReservoir:
    """Fixed Izhikevich reservoir with a vectorized, batched simulation."""

    def __init__(self,
                 N=200,
                 N_in=24,
                 recurrent=True,
                 topology='random',
                 exc_frac=0.8,
                 conn_prob=0.1,
                 input_conn_prob=0.3,
                 input_scale=10.0,
                 rec_scale=1.0,
                 noise_scale=3.0,
                 exc_type='RS',
                 inh_type='LTS',
                 seed=0):
        self.N = N
        self.N_in = N_in
        self.recurrent = recurrent
        self.topology = topology
        self.conn_prob = conn_prob
        self.noise_scale = noise_scale
        self.rec_scale = rec_scale
        self._noise_seed = seed + 999983       # separate stream for simulation noise

        # Two RNG streams: one for the neurons + input projection (so those are
        # IDENTICAL across topologies for a given seed), one for the recurrent
        # wiring (so only the topology differs in the comparison).
        rng = np.random.default_rng(seed)
        conn_rng = np.random.default_rng(seed + 4242)

        self.Ne = int(round(N * exc_frac))
        self.Ni = N - self.Ne
        Ne, Ni = self.Ne, self.Ni

        a_e, b_e, c_e, d_e, _ = NEURON_MODELS[exc_type]
        a_i, b_i, c_i, d_i, _ = NEURON_MODELS[inh_type]

        # Heterogeneous neuron parameters (same scheme as main.setup_network).
        re = rng.random((Ne, 1))
        ri = rng.random((Ni, 1))
        self.a = np.vstack([a_e * np.ones((Ne, 1)), a_i + 0.08 * ri])
        self.b = np.vstack([b_e * np.ones((Ne, 1)), b_i - 0.05 * ri])
        self.c = np.vstack([c_e + 15 * re ** 2, c_i * np.ones((Ni, 1))])
        self.d = np.vstack([d_e - 6 * re ** 2, d_i * np.ones((Ni, 1))])

        # Input connectivity (drawn before the topology so it is identical across
        # topologies for a given seed - only the recurrent wiring should change).
        W_in = input_scale * rng.standard_normal((N, N_in))
        W_in *= (rng.random((N, N_in)) < input_conn_prob)
        self.W_in = W_in

        # Recurrent connectivity: the topology picks WHERE synapses go; weights are
        # then signed (exc columns +, inh columns -). Stored at unit gain so
        # rec_scale / calibrate_rec_scale can rescale without changing structure.
        A = make_adjacency(topology, N, conn_prob, conn_rng)
        mags = conn_rng.random((N, N))
        W_unit = np.zeros((N, N))
        W_unit[:, :Ne] = 0.5 * mags[:, :Ne]
        W_unit[:, Ne:] = -1.0 * mags[:, Ne:]
        W_unit *= A
        np.fill_diagonal(W_unit, 0.0)
        self._W_unit = W_unit
        self.W_rec = rec_scale * W_unit if recurrent else np.zeros((N, N))

    def transform(self, X, return_rates_hz=False):
        """
        Run the reservoir on a batch of trials and return time-averaged features.

        Args:
            X: float32 array [B, N_in, T] of input spikes.
            return_rates_hz: if True, also return the mean per-neuron firing rate
                             in Hz (assumes 1 ms timesteps) for diagnostics.

        Returns:
            features: array [B, N] of mean spikes-per-timestep per neuron.
            (optionally) mean_rate_hz: scalar mean firing rate across the batch.
        """
        X = np.asarray(X, dtype=np.float64)
        B, N_in, T = X.shape
        if N_in != self.N_in:
            raise ValueError(f"X has N_in={N_in}, reservoir expects {self.N_in}.")

        rng = np.random.default_rng(self._noise_seed)
        Xt = np.transpose(X, (2, 1, 0))        # [T, N_in, B] for per-step matmul

        v = -65.0 * np.ones((self.N, B))
        u = self.b * v
        counts = np.zeros((self.N, B))
        fired_prev = np.zeros((self.N, B))

        for t in range(T):
            I = self.noise_scale * rng.standard_normal((self.N, B))
            I += self.W_in @ Xt[t]
            if self.recurrent:
                I += self.W_rec @ fired_prev
            # Two 0.5 ms half-steps for numerical stability (as in main.py).
            v += 0.5 * (0.04 * v * v + 5 * v + 140 - u + I)
            v += 0.5 * (0.04 * v * v + 5 * v + 140 - u + I)
            np.clip(v, -100.0, 30.0, out=v)    # guard against transient blow-up
            u += self.a * (self.b * v - u)

            fired = v >= 30.0
            counts += fired
            v = np.where(fired, self.c, v)
            u = np.where(fired, u + self.d, u)
            fired_prev = fired.astype(np.float64)

        features = (counts / T).T              # [B, N]
        if return_rates_hz:
            return features, float(counts.sum() / (self.N * B * T) * 1000.0)
        return features

    def calibrate_rec_scale(self, X_sample, target_hz,
                            candidates=(3, 5, 8, 12, 18, 26, 36, 50)):
        """
        Pick rec_scale so the reservoir's mean firing rate is near `target_hz`,
        without altering the topology. This lets different topologies be compared
        at a matched operating point, so accuracy differences reflect *structure*
        rather than activity level. No-op for a feedforward reservoir.

        Returns the chosen rec_scale.
        """
        if not self.recurrent:
            return self.rec_scale
        best = None
        for rs in candidates:
            self.W_rec = rs * self._W_unit
            _, hz = self.transform(X_sample, return_rates_hz=True)
            dist = abs(hz - target_hz)
            if best is None or dist < best[1]:
                best = (rs, dist)
        self.rec_scale = best[0]
        self.W_rec = best[0] * self._W_unit
        return best[0]

    def n_params(self):
        """Number of nonzero fixed weights (reservoirs are not trained)."""
        n = int(np.count_nonzero(self.W_in))
        if self.recurrent:
            n += int(np.count_nonzero(self.W_rec))
        return n
