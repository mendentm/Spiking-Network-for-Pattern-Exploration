# experiment_tracker.py

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import pickle

class ExperimentTracker:
    """
    Tracks and persists experiment results.
    Saves firing patterns, metrics, and metadata.
    """
    
    def __init__(self, experiment_name: str = "experiment"):
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = os.path.join("experiments", f"{experiment_name}_{timestamp}")
        
        # Create directory structure
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "firings"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "plots"), exist_ok=True)
        
        self.experiments = []
        self.experiment_counter = 0
        
        print(f"📁 Experiment tracker initialized: {self.base_dir}")
    
    def log_experiment(self, 
                      pattern_name: str, 
                      parameters: Dict[str, Any],
                      firings: np.ndarray,
                      metrics: Dict[str, Any]) -> int:
        """
        Log a single experiment.
        
        Args:
            pattern_name: Name of the pattern
            parameters: Pattern parameters used
            firings: Firing array [time, neuron_index]
            metrics: Calculated metrics
            
        Returns:
            int: Experiment ID
        """
        self.experiment_counter += 1
        exp_id = self.experiment_counter
        
        # Create experiment record
        experiment = {
            'id': exp_id,
            'timestamp': datetime.now().isoformat(),
            'pattern_name': pattern_name,
            'parameters': parameters,
            'metrics': metrics,
            'num_spikes': len(firings)
        }
        
        self.experiments.append(experiment)
        
        # Save firings data
        firings_path = os.path.join(self.base_dir, "firings", f"exp_{exp_id:04d}.npy")
        np.save(firings_path, firings)
        
        # Save metrics
        metrics_path = os.path.join(self.base_dir, "metrics", f"exp_{exp_id:04d}.json")
        with open(metrics_path, 'w') as f:
            json.dump(experiment, f, indent=2)
        
        return exp_id
    
    def load_firings(self, exp_id: int) -> np.ndarray:
        """Load firing data for a specific experiment."""
        firings_path = os.path.join(self.base_dir, "firings", f"exp_{exp_id:04d}.npy")
        return np.load(firings_path)
    
    def load_experiment(self, exp_id: int) -> Dict[str, Any]:
        """Load experiment metadata and metrics."""
        metrics_path = os.path.join(self.base_dir, "metrics", f"exp_{exp_id:04d}.json")
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def save_summary(self, additional_info: Dict[str, Any] = None):
        """Save a summary of all experiments."""
        summary = {
            'experiment_name': self.experiment_name,
            'total_experiments': self.experiment_counter,
            'timestamp': datetime.now().isoformat(),
            'experiments': self.experiments
        }
        
        if additional_info:
            summary.update(additional_info)
        
        summary_path = os.path.join(self.base_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Summary saved: {summary_path}")
        
        # Also save as pickle for easy loading
        pickle_path = os.path.join(self.base_dir, "summary.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(summary, f)
    
    def generate_report(self) -> str:
        """Generate a text report of all experiments."""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"EXPERIMENT REPORT: {self.experiment_name}")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Experiments: {self.experiment_counter}")
        report_lines.append("")
        
        # Pattern summary
        pattern_counts = {}
        for exp in self.experiments:
            name = exp['pattern_name']
            pattern_counts[name] = pattern_counts.get(name, 0) + 1
        
        report_lines.append("PATTERN DISTRIBUTION:")
        report_lines.append("-" * 80)
        for name, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            percentage = (count / self.experiment_counter) * 100
            report_lines.append(f"  {name:30s} {count:3d} experiments ({percentage:5.1f}%)")
        
        report_lines.append("")
        report_lines.append("METRIC STATISTICS:")
        report_lines.append("-" * 80)
        
        # Calculate metric statistics
        if self.experiments:
            metric_keys = list(self.experiments[0]['metrics'].keys())
            
            for key in metric_keys:
                values = [exp['metrics'][key] for exp in self.experiments]
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                report_lines.append(f"  {key:30s} {mean_val:8.3f} ± {std_val:6.3f} "
                                  f"[{min_val:8.3f}, {max_val:8.3f}]")
        
        report_lines.append("")
        report_lines.append("TOP EXPERIMENTS:")
        report_lines.append("-" * 80)
        
        # Find top experiments by different metrics
        key_metrics = ['synchrony_index', 'wave_score', 'mean_rate_total']
        
        for metric in key_metrics:
            if self.experiments:
                best_exp = max(self.experiments, key=lambda x: x['metrics'].get(metric, 0))
                report_lines.append(f"  Best {metric}:")
                report_lines.append(f"    Experiment #{best_exp['id']}: {best_exp['pattern_name']}")
                report_lines.append(f"    Value: {best_exp['metrics'][metric]:.4f}")
                report_lines.append("")
        
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(self.base_dir, "report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Report saved: {report_path}")
        
        return report_text
    
    def export_csv(self):
        """Export experiment data to CSV format."""
        import csv
        
        if not self.experiments:
            print("No experiments to export")
            return
        
        csv_path = os.path.join(self.base_dir, "experiments.csv")
        
        # Get all metric keys
        metric_keys = list(self.experiments[0]['metrics'].keys())
        
        # Create header
        header = ['id', 'timestamp', 'pattern_name', 'num_spikes'] + metric_keys
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for exp in self.experiments:
                row = [
                    exp['id'],
                    exp['timestamp'],
                    exp['pattern_name'],
                    exp['num_spikes']
                ]
                
                for key in metric_keys:
                    row.append(exp['metrics'].get(key, ''))
                
                writer.writerow(row)
        
        print(f"📊 CSV exported: {csv_path}")
    
    def get_experiments_by_pattern(self, pattern_name: str) -> List[Dict]:
        """Get all experiments for a specific pattern."""
        return [exp for exp in self.experiments if exp['pattern_name'] == pattern_name]
    
    def get_best_experiment(self, metric: str = 'synchrony_index') -> Dict:
        """Get the best experiment according to a metric."""
        if not self.experiments:
            return None
        
        return max(self.experiments, key=lambda x: x['metrics'].get(metric, 0))
    
    def compare_patterns(self, metric: str = 'synchrony_index') -> Dict[str, float]:
        """Compare average metric values across patterns."""
        pattern_metrics = {}
        
        for exp in self.experiments:
            pattern = exp['pattern_name']
            value = exp['metrics'].get(metric, 0)
            
            if pattern not in pattern_metrics:
                pattern_metrics[pattern] = []
            
            pattern_metrics[pattern].append(value)
        
        # Calculate averages
        pattern_averages = {
            pattern: np.mean(values)
            for pattern, values in pattern_metrics.items()
        }
        
        return dict(sorted(pattern_averages.items(), key=lambda x: -x[1]))
    
    def print_comparison(self, metric: str = 'synchrony_index'):
        """Print a comparison of patterns by metric."""
        comparison = self.compare_patterns(metric)
        
        print(f"\n{'='*60}")
        print(f"PATTERN COMPARISON: {metric}")
        print(f"{'='*60}")
        
        for i, (pattern, value) in enumerate(comparison.items(), 1):
            print(f"{i}. {pattern:30s} {value:8.4f}")
        
        print(f"{'='*60}\n")


def load_experiment_tracker(base_dir: str) -> ExperimentTracker:
    """Load a previously saved experiment tracker."""
    tracker = ExperimentTracker.__new__(ExperimentTracker)
    tracker.base_dir = base_dir
    
    # Load summary
    summary_path = os.path.join(base_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        tracker.experiment_name = summary.get('experiment_name', 'loaded')
        tracker.experiments = summary.get('experiments', [])
        tracker.experiment_counter = summary.get('total_experiments', 0)
    else:
        print(f"Warning: No summary found in {base_dir}")
        tracker.experiment_name = "loaded"
        tracker.experiments = []
        tracker.experiment_counter = 0
    
    return tracker


# Example usage
if __name__ == "__main__":
    # Create tracker
    tracker = ExperimentTracker("test_experiment")
    
    # Simulate logging some experiments
    for i in range(5):
        firings = np.random.rand(100, 2) * 1000
        metrics = {
            'mean_rate_total': np.random.rand() * 10,
            'synchrony_index': np.random.rand(),
            'wave_score': np.random.rand()
        }
        
        tracker.log_experiment(
            pattern_name=f"Pattern_{i % 3}",
            parameters={'param1': i},
            firings=firings,
            metrics=metrics
        )
    
    # Save summary and generate report
    tracker.save_summary()
    tracker.generate_report()
    tracker.export_csv()
    
    # Print comparison
    tracker.print_comparison('synchrony_index')