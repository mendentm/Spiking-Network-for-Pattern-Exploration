# neuron_models.py
"""
Shared Izhikevich neuron presets.
=================================
Each entry is [a, b, c, d, name] for the Izhikevich model:

    v' = 0.04*v^2 + 5*v + 140 - u + I
    u' = a*(b*v - u)
    if v >= 30:  v <- c,  u <- u + d

Kept in one place so the recurrent simulator (`main.py`) and the competition
reservoirs (`competition/`) draw from the same definitions (DRY).
"""

NEURON_MODELS = {
    'RS': [0.02, 0.2, -65, 8, "Regular Spiking"],
    'FS': [0.1, 0.2, -65, 2, "Fast Spiking"],
    'LTS': [0.02, 0.25, -65, 2, "Low-Threshold Spiking"],
    'IB': [0.02, 0.2, -55, 4, "Intrinsically Bursting"],
    'CH': [0.02, 0.2, -50, 2, "Chattering"],
}
