# Spiking Neural Network Simulation with Autonomous Pattern Exploration

An advanced simulation framework for spiking neural networks (SNNs) using the Izhikevich neuron model, featuring autonomous pattern exploration, comprehensive metrics analysis, and interactive 3D visualization.

## 🚀 New Features

### Autonomous Agent System
- **4 Exploration Strategies**: Random, Sequential, Metric Optimization, and Diversity Maximization
- **Automatic Pattern Discovery**: Systematically explores pattern space
- **Experiment Tracking**: Saves all results with metrics and metadata
- **Comparison Tools**: Analyze and compare pattern effectiveness

### Pattern Library
- **8+ Pattern Types**: Traveling waves, synchronized bursts, spirals, Game of Life, ripples, clusters, checkerboard, and random activity
- **Extensible Framework**: Easy to add custom patterns
- **Parameter Validation**: Automatic input checking

### Comprehensive Metrics
- **Firing Rate Analysis**: Per-population statistics with CV
- **Synchrony Measures**: Burst detection, participation ratio, ISI analysis
- **Spatial Coherence**: Wave detection, spatial correlation
- **Export Options**: JSON, CSV, and pickle formats

### Performance Optimizations
- **Sparse Matrices**: Efficient connectivity representation using scipy.sparse
- **Pre-allocated Arrays**: Reduced memory allocation overhead
- **Progress Bars**: Visual feedback with tqdm
- **Downsampling**: Handle large datasets in visualizations

## 📦 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "RSNN System"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install numpy matplotlib scipy plotly tqdm pandas

RSNN-System/
├── main.py                # Main control center
├── pattern_generator.py   # Pattern generation
├── pattern_library.py     # Pattern registry
├── autonomous_agent.py    # Exploration strategies
├── experiment_tracker.py  # Data persistence
├── metrics.py            # Analysis tools
├── plot_3d_raster.py     # Visualization
├── game_of_life.py       # Game of Life pattern
├── requirements.txt      # Dependencies
└── raster_plots/         # Output directory


📊 Metrics Explained
Firing Rate Metrics
Mean Rate: Average spikes per neuron per second
Coefficient of Variation (CV): Variability in firing rates
Synchrony Metrics
Synchrony Index: Variance-to-mean ratio of population activity
Burst Frequency: Rate of high-activity events
Participation Ratio: Fraction of active neurons
ISI CV: Regularity of inter-spike intervals
Spatial Metrics
Spatial Correlation: Correlation between adjacent neurons
Wave Score: Measure of traveling wave presence
Activity Sparsity: Fraction of active spatiotemporal bins

🧠 Network Architecture
Excitatory Neurons: Regular Spiking (RS) cells
Inhibitory Neurons: Low-Threshold Spiking (LTS) cells
Connectivity: Sparse random (default 10%)
Synaptic Weights: Excitatory (+0.5), Inhibitory (-1.0)