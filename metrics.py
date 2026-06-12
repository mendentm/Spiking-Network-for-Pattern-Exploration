# metrics.py

import numpy as np
from scipy import signal, stats
from collections import defaultdict

def calculate_firing_rate(firings, Ne, Ni, T, time_window=None):
    """
    Calculate mean firing rate for excitatory and inhibitory populations.
    
    Args:
        firings: Array of [time, neuron_index]
        Ne: Number of excitatory neurons
        Ni: Number of inhibitory neurons
        T: Total simulation time
        time_window: Optional tuple (start, end) to analyze specific window
        
    Returns:
        dict with firing rate statistics
    """
    if firings.size == 0:
        return {
            'mean_rate_exc': 0.0,
            'mean_rate_inh': 0.0,
            'mean_rate_total': 0.0,
            'std_rate_exc': 0.0,
            'std_rate_inh': 0.0
        }
    
    if time_window:
        mask = (firings[:, 0] >= time_window[0]) & (firings[:, 0] < time_window[1])
        firings = firings[mask]
        duration = time_window[1] - time_window[0]
    else:
        duration = T
    
    # Separate excitatory and inhibitory spikes
    exc_spikes = firings[firings[:, 1] < Ne]
    inh_spikes = firings[firings[:, 1] >= Ne]
    
    # Calculate rates (spikes per neuron per second)
    exc_rate = (len(exc_spikes) / Ne / duration) * 1000 if Ne > 0 else 0
    inh_rate = (len(inh_spikes) / Ni / duration) * 1000 if Ni > 0 else 0
    total_rate = (len(firings) / (Ne + Ni) / duration) * 1000
    
    # Calculate per-neuron firing rates for std (vectorized: bincount is O(spikes)
    # instead of the old O(neurons x spikes) double loop).
    if len(exc_spikes) > 0 and Ne > 0:
        exc_counts = np.bincount(exc_spikes[:, 1].astype(int), minlength=Ne)[:Ne]
    else:
        exc_counts = np.zeros(Ne)
    if len(inh_spikes) > 0 and Ni > 0:
        inh_counts = np.bincount(inh_spikes[:, 1].astype(int) - Ne, minlength=Ni)[:Ni]
    else:
        inh_counts = np.zeros(Ni)

    exc_neuron_rates = exc_counts / duration * 1000
    inh_neuron_rates = inh_counts / duration * 1000
    
    return {
        'mean_rate_exc': exc_rate,
        'mean_rate_inh': inh_rate,
        'mean_rate_total': total_rate,
        'std_rate_exc': np.std(exc_neuron_rates),
        'std_rate_inh': np.std(inh_neuron_rates),
        'cv_exc': np.std(exc_neuron_rates) / exc_rate if exc_rate > 0 else 0,
        'cv_inh': np.std(inh_neuron_rates) / inh_rate if inh_rate > 0 else 0
    }


def calculate_synchrony(firings, Ne, Ni, T, bin_size=10):
    """
    Calculate network synchrony using spike time tiling coefficient (STTC).
    Also computes simpler synchrony measures.
    
    Args:
        firings: Array of [time, neuron_index]
        Ne: Number of excitatory neurons
        Ni: Number of inhibitory neurons
        T: Total simulation time
        bin_size: Time bin size in ms for binning spikes
        
    Returns:
        dict with synchrony metrics
    """
    if firings.size == 0:
        return {
            'synchrony_index': 0.0,
            'burst_frequency': 0.0,
            'participation_ratio': 0.0
        }
    
    # Create binned spike counts
    n_bins = int(T / bin_size)
    spike_counts = np.zeros(n_bins)
    
    for spike_time in firings[:, 0]:
        bin_idx = int(spike_time / bin_size)
        if bin_idx < n_bins:
            spike_counts[bin_idx] += 1
    
    # Synchrony index: variance of population activity
    mean_count = np.mean(spike_counts)
    if mean_count > 0:
        synchrony_index = np.var(spike_counts) / mean_count
    else:
        synchrony_index = 0.0
    
    # Detect bursts (bins with activity > 2 std above mean)
    threshold = mean_count + 2 * np.std(spike_counts)
    bursts = spike_counts > threshold
    burst_frequency = (np.sum(bursts) / T) * 1000  # Hz
    
    # Participation ratio: fraction of neurons active
    active_neurons = len(np.unique(firings[:, 1]))
    participation_ratio = active_neurons / (Ne + Ni)
    
    # Calculate coefficient of variation of ISI
    isi_cv = calculate_isi_cv(firings)
    
    return {
        'synchrony_index': synchrony_index,
        'burst_frequency': burst_frequency,
        'participation_ratio': participation_ratio,
        'isi_cv': isi_cv,
        'mean_population_rate': mean_count / bin_size * 1000
    }


def calculate_isi_cv(firings):
    """Calculate coefficient of variation of inter-spike intervals."""
    if len(firings) < 2:
        return 0.0
    
    # Sort by time
    sorted_firings = firings[firings[:, 0].argsort()]
    
    # Calculate ISIs
    isis = np.diff(sorted_firings[:, 0])
    
    if len(isis) == 0 or np.mean(isis) == 0:
        return 0.0
    
    return np.std(isis) / np.mean(isis)


def calculate_spatial_coherence(firings, Ne, Ni, T):
    """
    Calculate spatial coherence of activity patterns.
    Measures how spatially organized the activity is.
    
    Args:
        firings: Array of [time, neuron_index]
        Ne: Number of excitatory neurons
        Ni: Number of inhibitory neurons
        T: Total simulation time
        
    Returns:
        dict with spatial metrics
    """
    if firings.size == 0:
        return {
            'spatial_correlation': 0.0,
            'wave_score': 0.0
        }
    
    total_neurons = Ne + Ni
    grid_side = int(np.ceil(np.sqrt(total_neurons)))
    
    # Create spatiotemporal activity matrix
    time_bins = min(100, T // 10)  # Limit to 100 bins for efficiency
    bin_size = T / time_bins
    activity_matrix = np.zeros((time_bins, total_neurons))
    
    for spike_time, neuron_idx in firings:
        time_bin = int(spike_time / bin_size)
        if time_bin < time_bins and neuron_idx < total_neurons:
            activity_matrix[time_bin, int(neuron_idx)] += 1
    
    # Calculate spatial correlation (average correlation between adjacent neurons).
    # One correlation per adjacent pair over the full time axis; silent pairs
    # (zero variance) are skipped because their correlation is undefined.
    spatial_corrs = []
    for n in range(total_neurons - 1):
        col_a = activity_matrix[:, n]
        col_b = activity_matrix[:, n + 1]
        if col_a.any() or col_b.any():
            corr = np.corrcoef(col_a, col_b)[0, 1]
            if not np.isnan(corr):
                spatial_corrs.append(corr)

    spatial_correlation = np.mean(spatial_corrs) if spatial_corrs else 0.0
    
    # Wave score: measure of traveling wave-like activity
    wave_score = calculate_wave_score(activity_matrix, grid_side)
    
    return {
        'spatial_correlation': spatial_correlation,
        'wave_score': wave_score,
        'activity_sparsity': np.sum(activity_matrix > 0) / activity_matrix.size
    }


def calculate_wave_score(activity_matrix, grid_side):
    """
    Calculate a score indicating presence of traveling waves.
    Higher score = more wave-like activity.
    """
    if activity_matrix.size == 0:
        return 0.0
    
    # Reshape to spatial grid
    time_bins, total_neurons = activity_matrix.shape
    
    # Calculate center of mass over time
    centers = []
    for t in range(time_bins):
        active = activity_matrix[t, :]
        if np.sum(active) > 0:
            # Map to 2D coordinates
            indices = np.arange(total_neurons)
            x_coords = indices % grid_side
            y_coords = indices // grid_side
            
            cx = np.sum(x_coords * active) / np.sum(active)
            cy = np.sum(y_coords * active) / np.sum(active)
            centers.append([cx, cy])
    
    if len(centers) < 2:
        return 0.0
    
    centers = np.array(centers)
    
    # Calculate smoothness of trajectory (lower variance in velocity = more wave-like)
    velocities = np.diff(centers, axis=0)
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    
    if len(velocity_magnitudes) == 0:
        return 0.0
    
    # Wave score: inverse of velocity variance (normalized)
    velocity_cv = np.std(velocity_magnitudes) / (np.mean(velocity_magnitudes) + 1e-10)
    wave_score = 1.0 / (1.0 + velocity_cv)
    
    return wave_score


def calculate_all_metrics(firings, Ne, Ni, T):
    """
    Calculate all available metrics for a firing pattern.
    
    Args:
        firings: Array of [time, neuron_index]
        Ne: Number of excitatory neurons
        Ni: Number of inhibitory neurons
        T: Total simulation time
        
    Returns:
        dict with all metrics
    """
    metrics = {}
    
    # Firing rate metrics
    metrics.update(calculate_firing_rate(firings, Ne, Ni, T))
    
    # Synchrony metrics
    metrics.update(calculate_synchrony(firings, Ne, Ni, T))
    
    # Spatial metrics
    metrics.update(calculate_spatial_coherence(firings, Ne, Ni, T))
    
    # Additional simple metrics
    metrics['total_spikes'] = len(firings)
    metrics['duration'] = T
    
    return metrics


def print_metrics_summary(metrics):
    """Pretty print metrics summary."""
    print("\n" + "="*60)
    print("NETWORK ACTIVITY METRICS")
    print("="*60)
    
    print("\n📊 FIRING RATES:")
    print(f"  Excitatory:  {metrics['mean_rate_exc']:.2f} ± {metrics['std_rate_exc']:.2f} Hz")
    print(f"  Inhibitory:  {metrics['mean_rate_inh']:.2f} ± {metrics['std_rate_inh']:.2f} Hz")
    print(f"  Total:       {metrics['mean_rate_total']:.2f} Hz")
    
    print("\n🔄 SYNCHRONY:")
    print(f"  Synchrony Index:      {metrics['synchrony_index']:.3f}")
    print(f"  Burst Frequency:      {metrics['burst_frequency']:.2f} Hz")
    print(f"  Participation Ratio:  {metrics['participation_ratio']:.2%}")
    print(f"  ISI CV:               {metrics['isi_cv']:.3f}")
    
    print("\n🌊 SPATIAL ORGANIZATION:")
    print(f"  Spatial Correlation:  {metrics['spatial_correlation']:.3f}")
    print(f"  Wave Score:           {metrics['wave_score']:.3f}")
    print(f"  Activity Sparsity:    {metrics['activity_sparsity']:.2%}")
    
    print("\n📈 SUMMARY:")
    print(f"  Total Spikes:  {metrics['total_spikes']}")
    print(f"  Duration:      {metrics['duration']} ms")
    print("="*60 + "\n")