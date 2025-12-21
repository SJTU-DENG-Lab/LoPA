"""
Performance Statistics Module: Used for collecting and analyzing performance data during the generation process.

Features:
1. Dataset-level statistics for peak TPS (tokens per second)
2. Dataset-level statistics for TPF (tokens per forward/step) for each sample at each step
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

eval_logger = logging.getLogger(__name__)


class GenerationStatsCollector:
    """Collects statistics during the generation process"""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir
        self.sample_stats: List[Dict[str, Any]] = []
        self.dataset_stats: Dict[str, Any] = {}
        
    def record_sample(
        self,
        sample_idx: int,
        tokens: int,
        steps: int,
        generation_time: float,
        tpf_per_step: Optional[List[float]] = None,
        dataset_name: Optional[str] = None,
    ):
        """Records statistics for a single sample
        
        Args:
            sample_idx: Sample index
            tokens: Number of tokens generated
            steps: Number of generation steps
            generation_time: Generation time (seconds)
            tpf_per_step: List of TPF (tokens per forward) for each step
            dataset_name: Dataset name
        """
        tps = tokens / generation_time if generation_time > 0 else 0.0
        avg_tpf = tokens / steps if steps > 0 else 0.0
        
        sample_stat = {
            "sample_idx": sample_idx,
            "tokens": tokens,
            "steps": steps,
            "generation_time": generation_time,
            "tps": tps,
            "avg_tpf": avg_tpf,
            "tpf_per_step": tpf_per_step or [],
            "dataset_name": dataset_name,
        }
        self.sample_stats.append(sample_stat)
        
    def compute_dataset_stats(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """Computes dataset-level statistics
        
        Args:
            dataset_name: Dataset name, if None statistics for all samples are computed
            
        Returns:
            Dictionary containing statistics like peak TPS
        """
        if dataset_name:
            relevant_samples = [s for s in self.sample_stats if s.get("dataset_name") == dataset_name]
        else:
            relevant_samples = self.sample_stats
            
        if not relevant_samples:
            return {}
            
        tps_list = [s["tps"] for s in relevant_samples]
        peak_tps = max(tps_list) if tps_list else 0.0
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        min_tps = min(tps_list) if tps_list else 0.0
        
        # Calculate mean TPS for top 10 fastest samples
        top10_tps_mean = 0.0
        if tps_list:
            sorted_tps = sorted(tps_list, reverse=True)
            top10_count = min(10, len(sorted_tps))
            top10_tps_mean = sum(sorted_tps[:top10_count]) / top10_count if top10_count > 0 else 0.0
        
        # Find information for the fastest sample
        peak_sample = None
        if relevant_samples:
            peak_sample_idx = max(range(len(relevant_samples)), key=lambda i: relevant_samples[i]["tps"])
            peak_sample = {
                "sample_idx": relevant_samples[peak_sample_idx]["sample_idx"],
                "tps": relevant_samples[peak_sample_idx]["tps"],
                "tokens": relevant_samples[peak_sample_idx]["tokens"],
                "steps": relevant_samples[peak_sample_idx]["steps"],
                "generation_time": relevant_samples[peak_sample_idx]["generation_time"],
            }
        
        total_tokens = sum(s["tokens"] for s in relevant_samples)
        total_steps = sum(s["steps"] for s in relevant_samples)
        total_time = sum(s["generation_time"] for s in relevant_samples)
        
        overall_tps = total_tokens / total_time if total_time > 0 else 0.0
        overall_tpf = total_tokens / total_steps if total_steps > 0 else 0.0
        
        stats = {
            "dataset_name": dataset_name or "all",
            "num_samples": len(relevant_samples),
            "peak_tps": peak_tps,
            "peak_sample": peak_sample,  # Details of the fastest sample
            "top10_tps_mean": top10_tps_mean,  # Mean TPS of the top 10 fastest samples
            "avg_tps": avg_tps,
            "min_tps": min_tps,
            "overall_tps": overall_tps,
            "overall_tpf": overall_tpf,
            "total_tokens": total_tokens,
            "total_steps": total_steps,
            "total_time": total_time,
        }
        
        if dataset_name:
            self.dataset_stats[dataset_name] = stats
        else:
            self.dataset_stats["all"] = stats
            
        return stats
    
    def build_tpf_table(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Builds a TPF table for each sample at each step
        
        Args:
            dataset_name: Dataset name, if None all samples are included
            
        Returns:
            TPF table, each row contains sample index, step number, and corresponding TPF
        """
        if dataset_name:
            relevant_samples = [s for s in self.sample_stats if s.get("dataset_name") == dataset_name]
        else:
            relevant_samples = self.sample_stats
            
        tpf_table = []
        for sample in relevant_samples:
            sample_idx = sample["sample_idx"]
            tpf_per_step = sample.get("tpf_per_step", [])
            dataset_name_actual = sample.get("dataset_name", "unknown")
            
            for step_idx, tpf in enumerate(tpf_per_step):
                tpf_table.append({
                    "sample_idx": sample_idx,
                    "step": step_idx + 1,  # Start counting from 1
                    "tpf": tpf,
                    "dataset_name": dataset_name_actual,
                })
                
        return tpf_table
    
    def save_stats(self, output_path: Optional[str] = None):
        """Saves statistics to a file
        
        Args:
            output_path: Output file path, if None save_dir is used
        """
        if output_path is None:
            if self.save_dir is None:
                eval_logger.warning("No save_dir specified, skipping stats save")
                return
            os.makedirs(self.save_dir, exist_ok=True)
            timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
            output_path = os.path.join(self.save_dir, f"generation_stats_{timestamp}.json")
        
        # Calculate statistics for all datasets
        datasets = set(s.get("dataset_name") for s in self.sample_stats if s.get("dataset_name"))
        for dataset_name in datasets:
            self.compute_dataset_stats(dataset_name)
        self.compute_dataset_stats()  # Calculate overall statistics
        
        # Build TPF table
        tpf_table = self.build_tpf_table()
        
        output_data = {
            "sample_stats": self.sample_stats,
            "dataset_stats": self.dataset_stats,
            "tpf_table": tpf_table,
            "timestamp": time.time(),
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        eval_logger.info(f"Statistics saved to {output_path}")
        
        # Also save TPF table in CSV format (for easier analysis)
        if tpf_table:
            csv_path = output_path.replace(".json", "_tpf_table.csv")
            import csv
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["sample_idx", "step", "tpf", "dataset_name"])
                writer.writeheader()
                writer.writerows(tpf_table)
            eval_logger.info(f"TPF table saved to {csv_path}")


def load_stats_from_file(stats_file: str) -> Dict[str, Any]:
    """Loads statistics from a file
    
    Args:
        stats_file: Path to the statistics file
        
    Returns:
        Dictionary containing statistics
    """
    with open(stats_file, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_peak_tps(stats_data: Dict[str, Any], dataset_name: Optional[str] = None) -> Dict[str, float]:
    """Analyzes peak TPS
    
    Args:
        stats_data: Data loaded from load_stats_from_file
        dataset_name: Dataset name, if None all datasets are analyzed
        
    Returns:
        Dictionary containing peak TPS information
    """
    dataset_stats = stats_data.get("dataset_stats", {})
    
    if dataset_name:
        if dataset_name in dataset_stats:
            return {
                "dataset": dataset_name,
                "peak_tps": dataset_stats[dataset_name]["peak_tps"],
                "avg_tps": dataset_stats[dataset_name]["avg_tps"],
                "overall_tps": dataset_stats[dataset_name]["overall_tps"],
            }
        else:
            eval_logger.warning(f"Dataset {dataset_name} not found in stats")
            return {}
    else:
        # Returns peak TPS for all datasets
        result = {}
        for name, stats in dataset_stats.items():
            result[name] = {
                "peak_tps": stats["peak_tps"],
                "avg_tps": stats["avg_tps"],
                "overall_tps": stats["overall_tps"],
            }
        return result


def get_tpf_table(stats_data: Dict[str, Any], dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Gets TPF table
    
    Args:
        stats_data: Data loaded from load_stats_from_file
        dataset_name: Dataset name, if None returns all samples
        
    Returns:
        List of TPF table entries
    """
    tpf_table = stats_data.get("tpf_table", [])
    
    if dataset_name:
        return [row for row in tpf_table if row.get("dataset_name") == dataset_name]
    else:
        return tpf_table


if __name__ == "__main__":
    # Example usage
    collector = GenerationStatsCollector(save_dir="./stats_output")
    
    # Simulate some data
    collector.record_sample(
        sample_idx=0,
        tokens=100,
        steps=10,
        generation_time=2.5,
        tpf_per_step=[10.0, 12.0, 8.0, 11.0, 9.0, 10.0, 12.0, 8.0, 11.0, 9.0],
        dataset_name="test_dataset",
    )
    
    collector.record_sample(
        sample_idx=1,
        tokens=150,
        steps=15,
        generation_time=3.0,
        tpf_per_step=[10.0] * 15,
        dataset_name="test_dataset",
    )
    
    # Compute statistics
    stats = collector.compute_dataset_stats("test_dataset")
    print("Dataset stats:", json.dumps(stats, indent=2))
    
    # Save statistics
    collector.save_stats()

