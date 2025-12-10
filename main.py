# main.py
"""
RSNN System - Main Control Center
==================================
Configure all parameters at the top and run the entire pipeline.

Usage:
    python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
from plot_3d_raster import generate_3d_raster_plot, generate_3d_raster_plotly
from autonomous_agent import ExplorationStrategy, RandomExploration, SequentialExploration, MetricOptimization
from experiment_tracker import ExperimentTracker
from patterns import PatternLibrary, get_library
from autonomous_agent import create_agent
from metrics import calculate_all_metrics, print_metrics_summary
from patterns import (
    generate_traveling_wave, generate_synchronized_bursts, generate_spiral_wave,
    generate_random_activity, generate_localized_clusters, generate_ripple_pattern,
    generate_checkerboard_pattern, generate_game_of_life, get_library
)

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES TO CONTROL EVERYTHING
# ============================================================================

# --- EXECUTION MODE ---
RUN_MODE = 'compare_all'  # Options: 'simulation', 'pattern', 'autonomous', 'compare_all'

# --- NETWORK PARAMETERS ---
Ne = 100                    # Number of excitatory neurons
Ni = 50                     # Number of inhibitory neurons
T = 5000                    # Total simulation time (ms)
connection_prob = 0.1       # Synaptic connection probability (0.0 to 1.0)

# --- NEURON MODEL PARAMETERS ---
# Options: 'RS' (Regular Spiking), 'FS' (Fast Spiking), 'LTS' (Low-Threshold Spiking)
EXCITATORY_TYPE = 'RS'
INHIBITORY_TYPE = 'LTS'

# --- PATTERN GENERATION (when RUN_MODE = 'pattern') ---
PATTERN_NAME = 'traveling_wave'  # Options below
PATTERN_PARAMS = {
    # For traveling_wave:
    'wave_speed': 1.5,
    'wave_width': 50,
    
    # For synchronized_bursts:
    'frequency': 10,
    'participation': 0.8,
    
    # For spiral_wave:
    'spiral_arms': 2,
    'rotation_speed': 0.05,
    
    # For game_of_life:
    'initial_fill': 0.1,
    
    # For random_activity:
    'firing_probability': 0.01,
    
    # For localized_clusters:
    'num_clusters': 3,
    'cluster_size': 30,
    
    # For ripple_pattern:
    'ripple_speed': 0.5,
    
    # For checkerboard:
    'square_size': 5,
    'flip_interval': 100
}

# Available patterns:
# 'traveling_wave', 'synchronized_bursts', 'spiral_wave', 'game_of_life',
# 'random_activity', 'localized_clusters', 'ripple_pattern', 'checkerboard'

# --- AUTONOMOUS AGENT (when RUN_MODE = 'autonomous') ---
AGENT_STRATEGY = 'random'      # Options: 'random', 'sequential', 'optimize', 'diversity'
NUM_EXPERIMENTS = 10           # Number of patterns to test
EXPERIMENT_NAME = 'my_exploration'
TARGET_METRIC = 'synchrony_index'  # For 'optimize' strategy
TARGET_VALUE = 2.0                 # For 'optimize' strategy

# --- VISUALIZATION OPTIONS ---
GENERATE_2D_PLOT = True
GENERATE_3D_PLOT = True
USE_PLOTLY_3D = True           # True = interactive HTML, False = static PNG
SHOW_PLOTS = False             # Display plots in window (blocks execution)
PLOT_DOWNSAMPLE = None         # Downsample for large datasets (e.g., 10 = every 10th spike)

# --- METRICS & ANALYSIS ---
CALCULATE_METRICS = True
PRINT_METRICS = True
SAVE_METRICS_JSON = True

# --- DATA SAVING ---
SAVE_FIRINGS_DATA = True       # Save spike data as .npy
OUTPUT_DIR = 'raster_plots'
EXPERIMENT_DIR = 'experiments'

# Output subdirectories
PLOTS_2D_DIR = os.path.join(OUTPUT_DIR, '2D Plots')
PLOTS_3D_DIR = os.path.join(OUTPUT_DIR, '3D Plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'Metrics')
FIRINGS_DIR = os.path.join(OUTPUT_DIR, 'Firing Data')

# --- PERFORMANCE OPTIONS ---
SHOW_PROGRESS_BAR = True       # Show tqdm progress during simulation
USE_SPARSE_MATRICES = True     # Use scipy.sparse for connectivity (recommended)
RANDOM_SEED = None             # Set to integer for reproducibility, None for random

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

# Set random seed if specified
if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)
    print(f"🎲 Random seed set to: {RANDOM_SEED}")

# Import modules
from patterns import (
    generate_traveling_wave, generate_synchronized_bursts, generate_spiral_wave,
    generate_random_activity, generate_localized_clusters, generate_ripple_pattern,
    generate_checkerboard_pattern, generate_game_of_life, get_library
)
# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(PLOTS_2D_DIR, exist_ok=True)
os.makedirs(PLOTS_3D_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIRINGS_DIR, exist_ok=True)

# ============================================================================
# NEURON MODEL DEFINITIONS
# ============================================================================

NEURON_MODELS = {
    'RS': [0.02, 0.2, -65, 8, "Regular Spiking"],
    'FS': [0.1, 0.2, -65, 2, "Fast Spiking"],
    'LTS': [0.02, 0.25, -65, 2, "Low-Threshold Spiking"],
    'IB': [0.02, 0.2, -55, 4, "Intrinsically Bursting"],
    'CH': [0.02, 0.2, -50, 2, "Chattering"]
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def setup_network():
    """Initialize network parameters and connectivity."""
    print(f"\n{'='*70}")
    print("NETWORK SETUP")
    print(f"{'='*70}")
    print(f"Excitatory neurons: {Ne} ({EXCITATORY_TYPE})")
    print(f"Inhibitory neurons: {Ni} ({INHIBITORY_TYPE})")
    print(f"Total neurons: {Ne + Ni}")
    print(f"Connection probability: {connection_prob*100}%")
    print(f"Simulation time: {T} ms")
    
    # Get neuron parameters
    a_e, b_e, c_e, d_e, name_e = NEURON_MODELS[EXCITATORY_TYPE]
    a_i, b_i, c_i, d_i, name_i = NEURON_MODELS[INHIBITORY_TYPE]
    
    # Initialize neuron parameters with variability
    re = np.random.rand(Ne, 1)
    ri = np.random.rand(Ni, 1)
    a = np.vstack((a_e * np.ones((Ne, 1)), a_i + 0.08 * ri))
    b = np.vstack((b_e * np.ones((Ne, 1)), b_i - 0.05 * ri))
    c = np.vstack((c_e + 15 * re**2, c_i * np.ones((Ni, 1))))
    d = np.vstack((d_e - 6 * re**2, d_i * np.ones((Ni, 1))))
    
    # Build connectivity matrix
    N_total = Ne + Ni
    
    if USE_SPARSE_MATRICES:
        print("Building sparse connectivity matrix...")
        S = lil_matrix((N_total, N_total))
        for i in range(N_total):
            exc_targets = np.random.rand(Ne) < connection_prob
            S[i, :Ne] = 0.5 * np.random.rand(Ne) * exc_targets
            inh_targets = np.random.rand(Ni) < connection_prob
            S[i, Ne:] = -1 * np.random.rand(Ni) * inh_targets
        S = S.tocsr()
        print(f"✓ Sparse matrix: {S.nnz} connections ({S.nnz/(N_total**2)*100:.2f}% density)")
    else:
        print("Building dense connectivity matrix...")
        S_full = np.hstack((0.5 * np.random.rand(N_total, Ne), -1 * np.random.rand(N_total, Ni)))
        sparse_mask = (np.random.rand(N_total, N_total) < connection_prob)
        S = S_full * sparse_mask
        print(f"✓ Dense matrix created")
    
    return a, b, c, d, S, name_e, name_i


def run_simulation(a, b, c, d, S):
    """Run Izhikevich model simulation."""
    print(f"\n{'='*70}")
    print("RUNNING IZHIKEVICH SIMULATION")
    print(f"{'='*70}")
    
    N_total = Ne + Ni
    v = -65 * np.ones((N_total, 1))
    u = b * v
    firings = np.array([]).reshape(0, 2)
    firing_bins = {t: [] for t in range(-1, T)}
    
    iterator = tqdm(range(T), desc="Simulating") if SHOW_PROGRESS_BAR else range(T)
    
    for t in iterator:
        # Input current
        I = np.vstack((5 * np.random.randn(Ne, 1), 2 * np.random.randn(Ni, 1)))
        
        # Synaptic input
        if firing_bins[t - 1]:
            fired_indices = np.array(firing_bins[t - 1])
            if USE_SPARSE_MATRICES:
                I += S[:, fired_indices].sum(axis=1).reshape(-1, 1)
            else:
                I += np.sum(S[:, fired_indices], axis=1).reshape(-1, 1)
        
        # Euler integration
        v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
        v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
        u += a * (b * v - u)
        
        # Detect spikes
        fired = np.where(v >= 30)[0]
        if fired.size > 0:
            new_firings = np.hstack((t * np.ones((fired.size, 1)), fired.reshape(-1, 1)))
            firings = np.vstack((firings, new_firings))
            firing_bins[t] = fired.tolist()
            v[fired] = c[fired]
            u[fired] = u[fired] + d[fired]
    
    print(f"✓ Simulation complete: {len(firings)} spikes generated")
    return firings


def generate_pattern():
    """Generate pattern using pattern generator."""
    print(f"\n{'='*70}")
    print(f"GENERATING PATTERN: {PATTERN_NAME}")
    print(f"{'='*70}")
    
    pattern_functions = {
        'traveling_wave': generate_traveling_wave,
        'synchronized_bursts': generate_synchronized_bursts,
        'spiral_wave': generate_spiral_wave,
        'random_activity': generate_random_activity,
        'localized_clusters': generate_localized_clusters,
        'ripple_pattern': generate_ripple_pattern,
        'checkerboard': generate_checkerboard_pattern,
        'game_of_life': generate_game_of_life

    }
    
    if PATTERN_NAME not in pattern_functions:
        raise ValueError(f"Unknown pattern: {PATTERN_NAME}. Available: {list(pattern_functions.keys())}")
    
    func = pattern_functions[PATTERN_NAME]
    
    # Get relevant parameters for this pattern
    import inspect
    sig = inspect.signature(func)
    relevant_params = {k: v for k, v in PATTERN_PARAMS.items() if k in sig.parameters}
    
    print(f"Parameters: {relevant_params}")
    firings = func(Ne, Ni, T, **relevant_params)
    print(f"✓ Generated {len(firings)} spikes")
    
    return firings


def run_autonomous_exploration():
    """Run autonomous agent exploration."""
    print(f"\n{'='*70}")
    print(f"AUTONOMOUS EXPLORATION")
    print(f"{'='*70}")
    
    agent = create_agent(
        Ne=Ne,
        Ni=Ni,
        T=T,
        strategy_name=AGENT_STRATEGY,
        experiment_name=EXPERIMENT_NAME,
        target_metric=TARGET_METRIC if AGENT_STRATEGY == 'optimize' else None,
        target_value=TARGET_VALUE if AGENT_STRATEGY == 'optimize' else None
    )
    
    agent.run_exploration(num_experiments=NUM_EXPERIMENTS, verbose=True)
    agent.save_summary()
    
    return agent


def compare_all_patterns():
    """Compare all available patterns."""
    print(f"\n{'='*70}")
    print("COMPARING ALL PATTERNS")
    print(f"{'='*70}")
    
    from patterns import get_library
    library = get_library()
    
    results = []
    
    for pattern_name in library.list_patterns():
        print(f"\n--- Testing: {pattern_name} ---")
        pattern = library.get_pattern(pattern_name)
        
        try:
            firings = pattern.generate(Ne, Ni, T)
            if len(firings) > 0:
                metrics = calculate_all_metrics(firings, Ne, Ni, T)
                results.append({
                    'name': pattern_name,
                    'firings': firings,
                    'metrics': metrics
                })
                print(f"✓ {len(firings)} spikes, rate: {metrics['mean_rate_total']:.2f} Hz")
                
                # Generate plots for this pattern
                if GENERATE_3D_PLOT or GENERATE_2D_PLOT:
                    create_plots(firings, "", "", f" - {pattern_name}", pattern_name=pattern_name)
            else:
                print(f"⚠️  No spikes generated")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    
    for metric in ['synchrony_index', 'wave_score', 'mean_rate_total']:
        print(f"\n{metric}:")
        sorted_results = sorted(results, key=lambda x: x['metrics'][metric], reverse=True)
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {r['name']:25s} {r['metrics'][metric]:.4f}")
    
    return results


def create_plots(firings, name_e, name_i, title_suffix="", pattern_name=None):
    """Generate 2D and 3D plots."""
    if len(firings) == 0:
        print("⚠️  No spikes to plot")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Create pattern-specific subfolder for 2D plots
    if pattern_name:
        pattern_2d_dir = os.path.join(PLOTS_2D_DIR, pattern_name)
        os.makedirs(pattern_2d_dir, exist_ok=True)
    else:
        pattern_2d_dir = PLOTS_2D_DIR
    
    # 2. Create pattern-specific subfolder for 3D plots
    if pattern_name:
        pattern_3d_dir = os.path.join(PLOTS_3D_DIR, pattern_name)
        os.makedirs(pattern_3d_dir, exist_ok=True)
    else:
        pattern_3d_dir = PLOTS_3D_DIR
    
    # 2D Plot
    if GENERATE_2D_PLOT:
        print("\nGenerating 2D raster plot...")
        plt.figure(figsize=(12, 7))
        
        # Downsample if needed
        plot_firings = firings
        if PLOT_DOWNSAMPLE and len(firings) > PLOT_DOWNSAMPLE:
            indices = np.random.choice(len(firings), PLOT_DOWNSAMPLE, replace=False)
            plot_firings = firings[indices]
            print(f"  Downsampled to {len(plot_firings)} spikes")
        
        plt.scatter(plot_firings[:, 0], plot_firings[:, 1], s=3, c='k', marker='.', alpha=0.6)
        plt.axhline(y=Ne, color='r', linestyle='--', linewidth=1, label='Exc/Inh boundary')
        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Neuron Index', fontsize=12)
        plt.xlim([0, T])
        plt.ylim([0, Ne + Ni])
        plt.title(f'Raster Plot{title_suffix}', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        
    # 3. Update the save path for the 2D plot
        filename_2d = f"raster_2d_{timestamp}.png"
        # Changed this line to use the new pattern_2d_dir
        full_path_2d = os.path.join(pattern_2d_dir, filename_2d) 
        plt.savefig(full_path_2d, dpi=300, bbox_inches='tight')
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        
        print(f"✓ 2D plot saved: {full_path_2d}")
    
    # 3D Plot
    if GENERATE_3D_PLOT:
        print("Generating 3D plot...")
        
        # ✅ Handle downsampling for ALL plots here
        plot_firings = firings
        if PLOT_DOWNSAMPLE and len(firings) > PLOT_DOWNSAMPLE:
            indices = np.random.choice(len(firings), PLOT_DOWNSAMPLE, replace=False)
            plot_firings = firings[indices]
            # This print is now only needed once if downsampling occurs
            # print(f"  Downsampled to {len(plot_firings)} spikes")

        if USE_PLOTLY_3D:
            filename_3d = f"raster_3d_{timestamp}.html"
            full_path_3d = os.path.join(pattern_3d_dir, filename_3d)
            generate_3d_raster_plotly(
                plot_firings, Ne, Ni, T,       # ✅ Use the (potentially) downsampled data
                output_filename=full_path_3d,
                show_browser=SHOW_PLOTS        # ✅ The 'show_browser' argument will now work
            )
        else:
            filename_3d = f"raster_3d_{timestamp}.png"
            full_path_3d = os.path.join(pattern_3d_dir, filename_3d)
            generate_3d_raster_plot(
                firings, Ne, Ni, T,
                output_filename=full_path_3d,
                downsample=PLOT_DOWNSAMPLE
            )
        
        print(f"✓ 3D plot saved: {full_path_3d}")


def analyze_firings(firings):
    """Calculate and display metrics."""
    if not CALCULATE_METRICS or len(firings) == 0:
        return None
    
    print("\nCalculating metrics...")
    metrics = calculate_all_metrics(firings, Ne, Ni, T)
    
    if PRINT_METRICS:
        print_metrics_summary(metrics)
    
    if SAVE_METRICS_JSON:
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(OUTPUT_DIR, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved: {metrics_file}")
    
    return metrics


def save_firings(firings):
    """Save firing data to disk."""
    if not SAVE_FIRINGS_DATA or len(firings) == 0:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    firings_file = os.path.join(OUTPUT_DIR, f"firings_{timestamp}.npy")
    np.save(firings_file, firings)
    print(f"✓ Firings data saved: {firings_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("RSNN SYSTEM - MAIN CONTROL CENTER")
    print("="*70)
    print(f"Run Mode: {RUN_MODE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*70)
    
    firings = None
    name_e, name_i = "", ""
    
    if RUN_MODE == 'simulation':
        # Run full Izhikevich simulation
        a, b, c, d, S, name_e, name_i = setup_network()
        firings = run_simulation(a, b, c, d, S)
        create_plots(firings, name_e, name_i, f" - {name_e}/{name_i}", pattern_name="simulation")
        analyze_firings(firings)
        save_firings(firings)
        
    elif RUN_MODE == 'pattern':
        # Generate specific pattern
        firings = generate_pattern()
        create_plots(firings, "", "", f" - {PATTERN_NAME}", pattern_name=PATTERN_NAME)
        analyze_firings(firings)
        save_firings(firings)
        
    elif RUN_MODE == 'autonomous':
        # Run autonomous exploration
        agent = run_autonomous_exploration()
        print(f"\n✅ Exploration complete! Results in: {agent.tracker.base_dir}")
        
    elif RUN_MODE == 'compare_all':
        # Compare all patterns
        results = compare_all_patterns()
        print(f"\n✅ Comparison complete! Tested {len(results)} patterns")
        
    else:
        print(f"❌ Unknown RUN_MODE: {RUN_MODE}")
        print("Valid options: 'simulation', 'pattern', 'autonomous', 'compare_all'")
        return
    
    print(f"\n{'='*70}")
    print("✅ EXECUTION COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()