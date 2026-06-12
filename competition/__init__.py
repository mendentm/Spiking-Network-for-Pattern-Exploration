# competition/__init__.py
"""
Topology Competition
====================
A reservoir-computing benchmark that pits three model families against each
other on one shared task, to answer: *does network topology matter?*

Contestants
-----------
1. RSNN reservoir          - recurrent Izhikevich network (topology-rich)
2. Feedforward-SNN reservoir - same Izhikevich neurons, recurrence removed
                               (the topology control)
3. Traditional NN          - from-scratch NumPy MLP on the raw input

The two spiking reservoirs are read out by an identical linear classifier, so
the RSNN-vs-feedforward comparison isolates the effect of *recurrent topology*
and nothing else. The MLP is the conventional, non-spiking reference point.

See `competition/compete.py::run_competition` for the orchestration and
`run_competition.py` (repo root) for the config-at-top entry point.
"""

from .task import generate_temporal_dataset, train_test_split
from .reservoir import IzhikevichReservoir
from .classifiers import RidgeReadout, NumpyMLP
from .connectivity import make_adjacency, TOPOLOGIES
from .compete import run_competition
from .evolution import run_evolution, EvolutionEngine, NetworkProblem, PatternProblem

__all__ = [
    "generate_temporal_dataset",
    "train_test_split",
    "IzhikevichReservoir",
    "RidgeReadout",
    "NumpyMLP",
    "make_adjacency",
    "TOPOLOGIES",
    "run_competition",
    "run_evolution",
    "EvolutionEngine",
    "NetworkProblem",
    "PatternProblem",
]
