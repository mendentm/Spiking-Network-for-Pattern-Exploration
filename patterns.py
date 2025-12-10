# patterns.py
"""
Unified Pattern Generation System
==================================
Contains all pattern generators with both functional and class-based interfaces.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import inspect


# ============================================================================
# BASE CLASS SYSTEM
# ============================================================================

class PatternBase(ABC):
    """Abstract base class for all pattern generators."""
    
    def __init__(self, name: str, description: str, default_params: Dict[str, Any]):
        self.name = name
        self.description = description
        self.default_params = default_params
        self.last_params = None
        self.last_firings = None
    
    @abstractmethod
    def generate(self, Ne: int, Ni: int, T: int, **kwargs) -> np.ndarray:
        """Generate firing pattern."""
        pass
    
    def get_param_info(self) -> Dict[str, Any]:
        """Return information about pattern parameters."""
        return {
            'name': self.name,
            'description': self.description,
            'default_params': self.default_params
        }
    
    def validate_params(self, **kwargs) -> Dict[str, Any]:
        """Validate and merge parameters with defaults."""
        params = self.default_params.copy()
        params.update(kwargs)
        return params


# ============================================================================
# FUNCTIONAL PATTERN GENERATORS (Simple Interface)
# ============================================================================

def generate_traveling_wave(Ne, Ni, T, wave_speed=1.0, wave_width=50):
    """
    Generates a traveling wave of activity.
    
    Args:
        Ne (int): Number of excitatory neurons
        Ni (int): Number of inhibitory neurons
        T (int): Total simulation time in ms
        wave_speed (float): Speed of wave in neurons per ms
        wave_width (int): Number of neurons active at once
        
    Returns:
        np.array: Firings array [time, neuron_index]
    """
    total_neurons = Ne + Ni
    estimated_spikes = int(T * wave_width * 0.7)
    firings_array = np.zeros((estimated_spikes, 2))
    spike_count = 0

    for t in range(T):
        wave_center = int((t * wave_speed) % total_neurons)
        start_neuron = wave_center - wave_width // 2
        end_neuron = wave_center + wave_width // 2
        
        for i in range(start_neuron, end_neuron):
            neuron_idx = i % total_neurons
            if np.random.rand() < 0.7:
                if spike_count < estimated_spikes:
                    firings_array[spike_count] = [t, neuron_idx]
                    spike_count += 1
    
    return firings_array[:spike_count]


def generate_synchronized_bursts(Ne, Ni, T, frequency=10, participation=0.8):
    """
    Generates synchronized network-wide bursts.
    
    Args:
        Ne (int): Number of excitatory neurons
        Ni (int): Number of inhibitory neurons
        T (int): Total simulation time in ms
        frequency (float): Burst frequency in Hz
        participation (float): Fraction of neurons firing (0.0 to 1.0)
        
    Returns:
        np.array: Firings array [time, neuron_index]
    """
    if not 0 < participation <= 1.0:
        raise ValueError("participation must be between 0 and 1")
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    
    total_neurons = Ne + Ni
    firings_list = []
    burst_interval_ms = int(1000 / frequency)
    burst_duration = 5
    
    for t in range(T):
        if t % burst_interval_ms < burst_duration:
            active_neurons = np.random.choice(
                total_neurons, 
                size=int(total_neurons * participation), 
                replace=False
            )
            for neuron_idx in active_neurons:
                firings_list.append([t, neuron_idx])

    return np.array(firings_list) if firings_list else np.array([]).reshape(0, 2)


def generate_spiral_wave(Ne, Ni, T, rotation_speed=0.05, spiral_arms=2, wave_width=20):
    """
    Generates spiral wave patterns.
    
    Args:
        Ne (int): Number of excitatory neurons
        Ni (int): Number of inhibitory neurons
        T (int): Total simulation time in ms
        rotation_speed (float): Angular velocity of rotation
        spiral_arms (int): Number of spiral arms
        wave_width (int): Width of spiral arms
        
    Returns:
        np.array: Firings array [time, neuron_index]
    """
    total_neurons = Ne + Ni
    grid_side = int(np.ceil(np.sqrt(total_neurons)))
    firings_list = []
    
    for t in range(T):
        angle = t * rotation_speed
        for arm in range(spiral_arms):
            arm_angle = angle + (arm * 2 * np.pi / spiral_arms)
            for r in range(wave_width):
                radius = (t * 0.05 + r) % grid_side
                x = int(grid_side / 2 + radius * np.cos(arm_angle)) % grid_side
                y = int(grid_side / 2 + radius * np.sin(arm_angle)) % grid_side
                neuron_idx = y * grid_side + x
                if neuron_idx < total_neurons and np.random.rand() < 0.6:
                    firings_list.append([t, neuron_idx])
    
    return np.array(firings_list) if firings_list else np.array([]).reshape(0, 2)


def generate_game_of_life(Ne, Ni, T, initial_fill=0.2):
    """
    Generates pattern based on Conway's Game of Life.
    
    Args:
        Ne (int): Number of excitatory neurons
        Ni (int): Number of inhibitory neurons
        T (int): Total simulation time (GOL generations)
        initial_fill (float): Fraction of initially alive cells (0.0 to 1.0)
        
    Returns:
        np.array: Firings array [time, neuron_index]
    """
    total_neurons = Ne + Ni
    grid_side = int(np.ceil(np.sqrt(total_neurons)))
    grid = np.random.choice([0, 1], size=(grid_side, grid_side), 
                           p=[1 - initial_fill, initial_fill])
    firings_list = []

    for t in range(T):
        # Record firings
        alive_coords = np.argwhere(grid == 1)
        for y, x in alive_coords:
            neuron_idx = y * grid_side + x
            if neuron_idx < total_neurons:
                firings_list.append([t, neuron_idx])

        # Update grid (GOL rules)
        neighbors = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                    np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) +
                    np.roll(np.roll(grid, 1, axis=0), 1, axis=1) +
                    np.roll(np.roll(grid, 1, axis=0), -1, axis=1) +
                    np.roll(np.roll(grid, -1, axis=0), 1, axis=1) +
                    np.roll(np.roll(grid, -1, axis=0), -1, axis=1))
        
        survives = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
        reproduces = (grid == 0) & (neighbors == 3)
        grid = np.zeros_like(grid)
        grid[survives | reproduces] = 1

    return np.array(firings_list) if firings_list else np.array([]).reshape(0, 2)


def generate_random_activity(Ne, Ni, T, firing_probability=0.01):
    """
    Generates random spatiotemporal activity.
    
    Args:
        Ne (int): Number of excitatory neurons
        Ni (int): Number of inhibitory neurons
        T (int): Total simulation time in ms
        firing_probability (float): Probability each neuron fires per timestep
        
    Returns:
        np.array: Firings array [time, neuron_index]
    """
    total_neurons = Ne + Ni
    firings_list = []
    
    for t in range(T):
        firing_neurons = np.random.rand(total_neurons) < firing_probability
        for neuron_idx in np.where(firing_neurons)[0]:
            firings_list.append([t, neuron_idx])
    
    return np.array(firings_list) if firings_list else np.array([]).reshape(0, 2)


def generate_localized_clusters(Ne, Ni, T, num_clusters=3, cluster_size=30, activity_rate=0.5):
    """
    Generates localized clusters of activity.
    
    Args:
        Ne (int): Number of excitatory neurons
        Ni (int): Number of inhibitory neurons
        T (int): Total simulation time in ms
        num_clusters (int): Number of active clusters
        cluster_size (int): Size of each cluster
        activity_rate (float): Firing probability within clusters
        
    Returns:
        np.array: Firings array [time, neuron_index]
    """
    total_neurons = Ne + Ni
    firings_list = []
    
    # Define cluster centers
    cluster_centers = np.random.choice(total_neurons, num_clusters, replace=False)
    
    for t in range(T):
        for center in cluster_centers:
            for offset in range(-cluster_size // 2, cluster_size // 2):
                neuron_idx = (center + offset) % total_neurons
                if np.random.rand() < activity_rate:
                    firings_list.append([t, neuron_idx])
    
    return np.array(firings_list) if firings_list else np.array([]).reshape(0, 2)


def generate_ripple_pattern(Ne, Ni, T, ripple_speed=0.5, frequency=5):
    """
    Generates concentric ripple patterns from center.
    
    Args:
        Ne (int): Number of excitatory neurons
        Ni (int): Number of inhibitory neurons
        T (int): Total simulation time in ms
        ripple_speed (float): Speed of ripple propagation
        frequency (int): Number of ripples to generate
        
    Returns:
        np.array: Firings array [time, neuron_index]
    """
    total_neurons = Ne + Ni
    grid_side = int(np.ceil(np.sqrt(total_neurons)))
    center_x, center_y = grid_side // 2, grid_side // 2
    firings_list = []
    
    ripple_interval = T // frequency
    
    for t in range(T):
        ripple_age = t % ripple_interval
        target_radius = ripple_age * ripple_speed
        
        for y in range(grid_side):
            for x in range(grid_side):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if abs(distance - target_radius) < 2 and np.random.rand() < 0.7:
                    neuron_idx = y * grid_side + x
                    if neuron_idx < total_neurons:
                        firings_list.append([t, neuron_idx])
    
    return np.array(firings_list) if firings_list else np.array([]).reshape(0, 2)


def generate_checkerboard_pattern(Ne, Ni, T, square_size=5, flip_interval=100):
    """
    Generates alternating checkerboard pattern.
    
    Args:
        Ne (int): Number of excitatory neurons
        Ni (int): Number of inhibitory neurons
        T (int): Total simulation time in ms
        square_size (int): Size of checkerboard squares
        flip_interval (int): Time between pattern flips
        
    Returns:
        np.array: Firings array [time, neuron_index]
    """
    total_neurons = Ne + Ni
    grid_side = int(np.ceil(np.sqrt(total_neurons)))
    firings_list = []
    
    for t in range(T):
        flip = (t // flip_interval) % 2
        for y in range(grid_side):
            for x in range(grid_side):
                is_white = ((x // square_size) + (y // square_size)) % 2
                if is_white == flip and np.random.rand() < 0.3:
                    neuron_idx = y * grid_side + x
                    if neuron_idx < total_neurons:
                        firings_list.append([t, neuron_idx])
    
    return np.array(firings_list) if firings_list else np.array([]).reshape(0, 2)


# Alias for backward compatibility
generate_gol_firings = generate_game_of_life


# ============================================================================
# CLASS-BASED PATTERN WRAPPERS
# ============================================================================

class TravelingWavePattern(PatternBase):
    """Traveling wave pattern class."""
    
    def __init__(self):
        super().__init__(
            name="Traveling Wave",
            description="A wave of activity that propagates through the network",
            default_params={'wave_speed': 1.0, 'wave_width': 50}
        )
    
    def generate(self, Ne: int, Ni: int, T: int, **kwargs) -> np.ndarray:
        params = self.validate_params(**kwargs)
        self.last_params = params
        self.last_firings = generate_traveling_wave(Ne, Ni, T, **params)
        return self.last_firings


class SynchronizedBurstPattern(PatternBase):
    """Synchronized burst pattern class."""
    
    def __init__(self):
        super().__init__(
            name="Synchronized Bursts",
            description="Network-wide synchronized bursting activity",
            default_params={'frequency': 10, 'participation': 0.8}
        )
    
    def generate(self, Ne: int, Ni: int, T: int, **kwargs) -> np.ndarray:
        params = self.validate_params(**kwargs)
        self.last_params = params
        self.last_firings = generate_synchronized_bursts(Ne, Ni, T, **params)
        return self.last_firings


class SpiralWavePattern(PatternBase):
    """Spiral wave pattern class."""
    
    def __init__(self):
        super().__init__(
            name="Spiral Wave",
            description="Rotating spiral waves of activity",
            default_params={'rotation_speed': 0.05, 'spiral_arms': 2, 'wave_width': 20}
        )
    
    def generate(self, Ne: int, Ni: int, T: int, **kwargs) -> np.ndarray:
        params = self.validate_params(**kwargs)
        self.last_params = params
        self.last_firings = generate_spiral_wave(Ne, Ni, T, **params)
        return self.last_firings


class GameOfLifePattern(PatternBase):
    """Conway's Game of Life pattern class."""
    
    def __init__(self):
        super().__init__(
            name="Game of Life",
            description="Conway's cellular automaton",
            default_params={'initial_fill': 0.2}
        )
    
    def generate(self, Ne: int, Ni: int, T: int, **kwargs) -> np.ndarray:
        params = self.validate_params(**kwargs)
        self.last_params = params
        self.last_firings = generate_game_of_life(Ne, Ni, T, **params)
        return self.last_firings


class RandomActivityPattern(PatternBase):
    """Random activity pattern class."""
    
    def __init__(self):
        super().__init__(
            name="Random Activity",
            description="Spatiotemporal random noise",
            default_params={'firing_probability': 0.01}
        )
    
    def generate(self, Ne: int, Ni: int, T: int, **kwargs) -> np.ndarray:
        params = self.validate_params(**kwargs)
        self.last_params = params
        self.last_firings = generate_random_activity(Ne, Ni, T, **params)
        return self.last_firings


class LocalizedClustersPattern(PatternBase):
    """Localized clusters pattern class."""
    
    def __init__(self):
        super().__init__(
            name="Localized Clusters",
            description="Spatially clustered activity",
            default_params={'num_clusters': 3, 'cluster_size': 30, 'activity_rate': 0.5}
        )
    
    def generate(self, Ne: int, Ni: int, T: int, **kwargs) -> np.ndarray:
        params = self.validate_params(**kwargs)
        self.last_params = params
        self.last_firings = generate_localized_clusters(Ne, Ni, T, **params)
        return self.last_firings


class RipplePattern(PatternBase):
    """Ripple pattern class."""
    
    def __init__(self):
        super().__init__(
            name="Ripple Pattern",
            description="Concentric waves from center",
            default_params={'ripple_speed': 0.5, 'frequency': 5}
        )
    
    def generate(self, Ne: int, Ni: int, T: int, **kwargs) -> np.ndarray:
        params = self.validate_params(**kwargs)
        self.last_params = params
        self.last_firings = generate_ripple_pattern(Ne, Ni, T, **params)
        return self.last_firings


class CheckerboardPattern(PatternBase):
    """Checkerboard pattern class."""
    
    def __init__(self):
        super().__init__(
            name="Checkerboard",
            description="Alternating spatial pattern",
            default_params={'square_size': 5, 'flip_interval': 100}
        )
    
    def generate(self, Ne: int, Ni: int, T: int, **kwargs) -> np.ndarray:
        params = self.validate_params(**kwargs)
        self.last_params = params
        self.last_firings = generate_checkerboard_pattern(Ne, Ni, T, **params)
        return self.last_firings


# ============================================================================
# PATTERN LIBRARY
# ============================================================================

class PatternLibrary:
    """Registry of all available patterns."""
    
    def __init__(self):
        self.patterns = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register all default patterns."""
        self.register(TravelingWavePattern())
        self.register(SynchronizedBurstPattern())
        self.register(SpiralWavePattern())
        self.register(GameOfLifePattern())
        self.register(RandomActivityPattern())
        self.register(LocalizedClustersPattern())
        self.register(RipplePattern())
        self.register(CheckerboardPattern())
    
    def register(self, pattern: PatternBase):
        """Register a new pattern."""
        self.patterns[pattern.name.lower().replace(' ', '_')] = pattern
    
    def get_pattern(self, name: str) -> PatternBase:
        """Get pattern by name."""
        key = name.lower().replace(' ', '_')
        if key not in self.patterns:
            raise ValueError(f"Pattern '{name}' not found. Available: {self.list_patterns()}")
        return self.patterns[key]
    
    def list_patterns(self) -> List[str]:
        """List all available pattern names."""
        return list(self.patterns.keys())
    
    def print_catalog(self):
        """Print catalog of all patterns."""
        print("\n" + "="*70)
        print("PATTERN CATALOG")
        print("="*70)
        for name, pattern in self.patterns.items():
            info = pattern.get_param_info()
            print(f"\n{info['name']}:")
            print(f"  {info['description']}")
            print(f"  Parameters: {info['default_params']}")
        print("="*70)


# Singleton instance
_library = None

def get_library() -> PatternLibrary:
    """Get the global pattern library instance."""
    global _library
    if _library is None:
        _library = PatternLibrary()
    return _library


def get_pattern_info(pattern_name: str) -> Dict[str, Any]:
    """Get information about a specific pattern."""
    library = get_library()
    pattern = library.get_pattern(pattern_name)
    return pattern.get_param_info()