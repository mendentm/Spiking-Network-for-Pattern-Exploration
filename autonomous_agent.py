# autonomous_agent.py

import numpy as np
from typing import Dict, List, Any, Optional
from metrics import calculate_all_metrics
from patterns import PatternLibrary, get_library
from experiment_tracker import ExperimentTracker
import random

class ExplorationStrategy:
    """Base class for exploration strategies."""
    
    def select_next_pattern(self, library, history: List[Dict]) -> tuple:
        """
        Select next pattern and parameters.
        
        Returns:
            tuple: (pattern_name, parameters_dict)
        """
        raise NotImplementedError


class RandomExploration(ExplorationStrategy):
    """Randomly explore all patterns with random parameters."""
    
    def __init__(self, param_variation=0.3):
        self.param_variation = param_variation
    
    def select_next_pattern(self, library, history: List[Dict]) -> tuple:
        # Pick random pattern
        pattern_names = library.list_patterns()
        pattern_name = random.choice(pattern_names)
        pattern = library.get_pattern(pattern_name)
        
        # Vary parameters randomly
        params = {}
        for key, default_value in pattern.default_params.items():
            if isinstance(default_value, (int, float)):
                variation = default_value * self.param_variation
                params[key] = default_value + random.uniform(-variation, variation)
                
                # Keep positive values
                if default_value > 0:
                    params[key] = max(0.01, params[key])
            else:
                params[key] = default_value
        
        return pattern_name, params


class SequentialExploration(ExplorationStrategy):
    """Systematically explore all patterns in order."""
    
    def __init__(self):
        self.current_index = 0
        self.pattern_names = None
    
    def select_next_pattern(self, library, history: List[Dict]) -> tuple:
        if self.pattern_names is None:
            self.pattern_names = library.list_patterns()
        
        pattern_name = self.pattern_names[self.current_index % len(self.pattern_names)]
        self.current_index += 1
        
        pattern = library.get_pattern(pattern_name)
        return pattern_name, pattern.default_params.copy()


class MetricOptimization(ExplorationStrategy):
    """Optimize for specific metric values."""
    
    def __init__(self, target_metric='synchrony_index', target_value=1.0, exploration_rate=0.3):
        self.target_metric = target_metric
        self.target_value = target_value
        self.exploration_rate = exploration_rate
        self.best_pattern = None
        self.best_params = None
        self.best_score = float('inf')
    
    def select_next_pattern(self, library, history: List[Dict]) -> tuple:
        # Update best based on history
        if history:
            last_exp = history[-1]
            if 'metrics' in last_exp:
                metric_value = last_exp['metrics'].get(self.target_metric, 0)
                score = abs(metric_value - self.target_value)
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_pattern = last_exp['pattern_name']
                    self.best_params = last_exp['parameters'].copy()
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate or self.best_pattern is None:
            # Explore: random pattern
            pattern_names = library.list_patterns()
            pattern_name = random.choice(pattern_names)
            pattern = library.get_pattern(pattern_name)
            params = pattern.default_params.copy()
        else:
            # Exploit: mutate best pattern
            pattern_name = self.best_pattern
            pattern = library.get_pattern(pattern_name)
            params = self._mutate_params(self.best_params, pattern.default_params)
        
        return pattern_name, params
    
    def _mutate_params(self, params: Dict, defaults: Dict) -> Dict:
        """Mutate parameters slightly."""
        mutated = params.copy()
        
        for key, value in mutated.items():
            if isinstance(value, (int, float)) and key in defaults:
                mutation = value * random.uniform(-0.2, 0.2)
                mutated[key] = value + mutation
                
                # Keep positive
                if defaults[key] > 0:
                    mutated[key] = max(0.01, mutated[key])
        
        return mutated


class DiversityMaximization(ExplorationStrategy):
    """Maximize diversity of explored patterns."""
    
    def __init__(self, diversity_weight=0.7):
        self.diversity_weight = diversity_weight
        self.explored_patterns = set()
    
    def select_next_pattern(self, library, history: List[Dict]) -> tuple:
        pattern_names = library.list_patterns()
        
        # Update explored set
        for exp in history:
            self.explored_patterns.add(exp['pattern_name'])
        
        # Prefer unexplored patterns
        unexplored = [p for p in pattern_names if p not in self.explored_patterns]
        
        if unexplored and random.random() < self.diversity_weight:
            pattern_name = random.choice(unexplored)
        else:
            pattern_name = random.choice(pattern_names)
        
        pattern = library.get_pattern(pattern_name)
        
        # Vary parameters
        params = {}
        for key, default_value in pattern.default_params.items():
            if isinstance(default_value, (int, float)):
                params[key] = default_value * random.uniform(0.5, 1.5)
                if default_value > 0:
                    params[key] = max(0.01, params[key])
            else:
                params[key] = default_value
        
        return pattern_name, params


class AutonomousAgent:
    """
    Autonomous agent that explores pattern space and learns.
    """
    
    def __init__(self, 
                 Ne: int, 
                 Ni: int, 
                 T: int,
                 strategy: Optional[ExplorationStrategy] = None,
                 experiment_name: str = "autonomous_exploration"):
        
        self.Ne = Ne
        self.Ni = Ni
        self.T = T
        self.library = get_library()
        self.strategy = strategy or RandomExploration()
        self.tracker = ExperimentTracker(experiment_name)
        self.history = []
        self.iteration = 0
        
        print(f"\n🤖 Autonomous Agent Initialized")
        print(f"   Network: {Ne} excitatory, {Ni} inhibitory neurons")
        print(f"   Duration: {T} ms")
        print(f"   Strategy: {self.strategy.__class__.__name__}")
        print(f"   Available patterns: {len(self.library.list_patterns())}")
    
    def run_single_experiment(self, verbose=True) -> Dict[str, Any]:
        """Run a single pattern experiment."""
        self.iteration += 1
        
        # Select pattern and parameters
        pattern_name, params = self.strategy.select_next_pattern(self.library, self.history)
        pattern = self.library.get_pattern(pattern_name)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔬 Experiment {self.iteration}: {pattern_name}")
            print(f"{'='*70}")
            print(f"Parameters: {params}")
        
        # Generate pattern
        firings = pattern.generate(self.Ne, self.Ni, self.T, **params)
        
        if verbose:
            print(f"Generated {len(firings)} spikes")
        
        # Calculate metrics
        metrics = calculate_all_metrics(firings, self.Ne, self.Ni, self.T)
        
        # Store experiment
        experiment = {
            'iteration': self.iteration,
            'pattern_name': pattern_name,
            'parameters': params,
            'metrics': metrics,
            'num_spikes': len(firings)
        }
        
        self.history.append(experiment)
        self.tracker.log_experiment(pattern_name, params, firings, metrics)
        
        if verbose:
            self._print_experiment_summary(metrics)
        
        return experiment
    
    def run_exploration(self, num_experiments: int, verbose=True):
        """Run multiple experiments."""
        print(f"\n🚀 Starting exploration: {num_experiments} experiments")
        print("="*70)
        
        for i in range(num_experiments):
            try:
                self.run_single_experiment(verbose=verbose)
            except Exception as e:
                print(f"❌ Error in experiment {i+1}: {e}")
                continue
        
        print(f"\n✅ Exploration complete!")
        self._print_exploration_summary()
    
    def _print_experiment_summary(self, metrics: Dict):
        """Print summary of single experiment."""
        print(f"\n📊 Quick Metrics:")
        print(f"   Firing Rate: {metrics['mean_rate_total']:.2f} Hz")
        print(f"   Synchrony: {metrics['synchrony_index']:.3f}")
        print(f"   Wave Score: {metrics['wave_score']:.3f}")
    
    def _print_exploration_summary(self):
        """Print summary of all experiments."""
        print(f"\n{'='*70}")
        print(f"📈 EXPLORATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total experiments: {len(self.history)}")
        
        # Pattern distribution
        pattern_counts = {}
        for exp in self.history:
            name = exp['pattern_name']
            pattern_counts[name] = pattern_counts.get(name, 0) + 1
        
        print(f"\n🎨 Pattern Distribution:")
        for name, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"   {name}: {count} experiments")
        
        # Best patterns by metric
        if self.history:
            print(f"\n🏆 Best Patterns:")
            
            # Highest synchrony
            best_sync = max(self.history, key=lambda x: x['metrics']['synchrony_index'])
            print(f"   Highest Synchrony: {best_sync['pattern_name']} "
                  f"(score: {best_sync['metrics']['synchrony_index']:.3f})")
            
            # Best wave
            best_wave = max(self.history, key=lambda x: x['metrics']['wave_score'])
            print(f"   Best Wave Pattern: {best_wave['pattern_name']} "
                  f"(score: {best_wave['metrics']['wave_score']:.3f})")
            
            # Highest firing rate
            best_rate = max(self.history, key=lambda x: x['metrics']['mean_rate_total'])
            print(f"   Highest Activity: {best_rate['pattern_name']} "
                  f"(rate: {best_rate['metrics']['mean_rate_total']:.2f} Hz)")
        
        print(f"\n💾 Results saved to: {self.tracker.base_dir}")
        print("="*70)
    
    def get_best_pattern(self, metric='synchrony_index') -> Dict:
        """Get the best pattern according to a metric."""
        if not self.history:
            return None
        
        return max(self.history, key=lambda x: x['metrics'].get(metric, 0))
    
    def get_pattern_statistics(self) -> Dict:
        """Get statistics across all experiments."""
        if not self.history:
            return {}
        
        stats = {}
        
        # Aggregate metrics
        all_metrics = [exp['metrics'] for exp in self.history]
        
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return stats
    
    def save_summary(self):
        """Save exploration summary."""
        self.tracker.save_summary({
            'total_experiments': len(self.history),
            'strategy': self.strategy.__class__.__name__,
            'network_config': {'Ne': self.Ne, 'Ni': self.Ni, 'T': self.T},
            'pattern_statistics': self.get_pattern_statistics()
        })


def create_agent(Ne=100, Ni=50, T=10000, strategy_name='random', **strategy_kwargs):
    """
    Factory function to create an autonomous agent.
    
    Args:
        Ne: Number of excitatory neurons
        Ni: Number of inhibitory neurons
        T: Simulation time
        strategy_name: 'random', 'sequential', 'optimize', or 'diversity'
        **strategy_kwargs: Additional arguments for the strategy
        
    Returns:
        AutonomousAgent instance
    """
    strategies = {
        'random': RandomExploration,
        'sequential': SequentialExploration,
        'optimize': MetricOptimization,
        'diversity': DiversityMaximization
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(strategies.keys())}")
    
    strategy_class = strategies[strategy_name]
    strategy = strategy_class(**strategy_kwargs)
    
    return AutonomousAgent(Ne, Ni, T, strategy=strategy)


# Example usage
if __name__ == "__main__":
    # Create agent with random exploration
    agent = create_agent(Ne=100, Ni=50, T=5000, strategy_name='random')
    
    # Run exploration
    agent.run_exploration(num_experiments=10, verbose=True)
    
    # Get best pattern
    best = agent.get_best_pattern('synchrony_index')
    print(f"\nBest pattern for synchrony: {best['pattern_name']}")
    
    # Save results
    agent.save_summary()