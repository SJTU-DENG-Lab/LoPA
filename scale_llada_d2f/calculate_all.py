#!/usr/bin/env python3
"""
遍历文件夹，找到包含 rank 统计文件的文件夹，
汇总 rank_0 到 rank_7 的统计数据，计算整体统计信息并保存到新的 json 文件中。
兼容旧文件名 rank_*_final_stats.json 和当前文件名 rank_*_stats.json。
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def find_rank_stats_folders(root_path: str) -> List[str]:
    """
    遍历文件夹，找到所有包含 rank_0 统计文件的文件夹
    
    Args:
        root_path: 根目录路径
        
    Returns:
        包含 rank_0 统计文件的文件夹路径列表
    """
    rank_folders = []
    
    for root, dirs, files in os.walk(root_path):
        if "rank_0_final_stats.json" in files or "rank_0_stats.json" in files:
            rank_folders.append(root)
    
    return rank_folders


def load_rank_stats(folder_path: str) -> Dict[str, Any]:
    """
    加载一个文件夹中所有rank文件的统计数据
    
    Args:
        folder_path: 包含rank文件的文件夹路径
        
    Returns:
        汇总的统计数据
    """
    total_stats = {
        "processed_samples": 0,
        "total_samples": 0,
        "total_generated_tokens_including_eos": 0,
        "total_actual_tokens_excluding_eos": 0,
        "total_parallel_steps": 0,
        "total_time": 0.0,
        "generated_tokens_including_eos_per_second_sum": 0.0,
        "actual_tokens_excluding_eos_per_second_sum": 0.0,
        "generated_tokens_including_eos_per_step_sum": 0.0,
        "actual_tokens_excluding_eos_per_step_sum": 0.0,
        "rank_files_found": [],
        "rank_files_missing": []
    }
    
    # 检查rank_0到rank_7的文件
    for rank in range(8):
        rank_file = None
        candidate_files = [
            os.path.join(folder_path, f"rank_{rank}_final_stats.json"),
            os.path.join(folder_path, f"rank_{rank}_stats.json"),
        ]
        for candidate in candidate_files:
            if os.path.exists(candidate):
                rank_file = candidate
                break
        
        if rank_file is not None:
            try:
                with open(rank_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 累加数据 - 适配当前保存结果中的字段名
                total_stats["processed_samples"] += data.get("processed_samples", 0)
                total_stats["total_samples"] += data.get("total_samples", 0)
                total_stats["total_generated_tokens_including_eos"] += data.get("total_generated_tokens_including_eos", 0)
                total_stats["total_actual_tokens_excluding_eos"] += data.get("total_actual_tokens_excluding_eos", 0)
                total_stats["total_parallel_steps"] += data.get("total_parallel_steps", 0)
                total_stats["total_time"] += data.get("total_time", 0.0)
                
                # 累加每个 rank 的 tps/tpf，后续计算跨 rank 平均值
                total_stats["generated_tokens_including_eos_per_second_sum"] += data.get("generated_tokens_including_eos_per_second", 0.0)
                total_stats["actual_tokens_excluding_eos_per_second_sum"] += data.get("actual_tokens_excluding_eos_per_second", 0.0)
                total_stats["generated_tokens_including_eos_per_step_sum"] += data.get("generated_tokens_including_eos_per_step", 0.0)
                total_stats["actual_tokens_excluding_eos_per_step_sum"] += data.get("actual_tokens_excluding_eos_per_step", 0.0)
                
                total_stats["rank_files_found"].append(os.path.basename(rank_file))
                
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Error reading {rank_file}: {e}")
                total_stats["rank_files_missing"].append(f"rank_{rank}")
        else:
            total_stats["rank_files_missing"].append(f"rank_{rank}")
    
    return total_stats


def calculate_summary_stats(total_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据汇总数据计算统计指标
    
    Args:
        total_stats: 汇总的统计数据
        
    Returns:
        包含计算指标的统计数据
    """
    summary = total_stats.copy()
    num_ranks_found = len(summary["rank_files_found"])
    
    # 计算两种 token 口径下的整体 tps
    if summary["total_time"] > 0:
        summary["overall_generated_tokens_including_eos_per_second"] = (
            summary["total_generated_tokens_including_eos"] / summary["total_time"]
        )
        summary["overall_actual_tokens_excluding_eos_per_second"] = (
            summary["total_actual_tokens_excluding_eos"] / summary["total_time"]
        )
    else:
        summary["overall_generated_tokens_including_eos_per_second"] = 0.0
        summary["overall_actual_tokens_excluding_eos_per_second"] = 0.0
    
    # 计算两种 token 口径下的跨 rank 平均 tps
    if num_ranks_found > 0:
        summary["avg_generated_tokens_including_eos_per_second"] = (
            summary["generated_tokens_including_eos_per_second_sum"] / num_ranks_found
        )
        summary["avg_actual_tokens_excluding_eos_per_second"] = (
            summary["actual_tokens_excluding_eos_per_second_sum"] / num_ranks_found
        )
    else:
        summary["avg_generated_tokens_including_eos_per_second"] = 0.0
        summary["avg_actual_tokens_excluding_eos_per_second"] = 0.0
    
    # 计算两种 token 口径下的跨 rank 平均 tpf
    if num_ranks_found > 0:
        summary["avg_generated_tokens_including_eos_per_step"] = (
            summary["generated_tokens_including_eos_per_step_sum"] / num_ranks_found
        )
        summary["avg_actual_tokens_excluding_eos_per_step"] = (
            summary["actual_tokens_excluding_eos_per_step_sum"] / num_ranks_found
        )
    else:
        summary["avg_generated_tokens_including_eos_per_step"] = 0.0
        summary["avg_actual_tokens_excluding_eos_per_step"] = 0.0
    
    # 计算两种 token 口径下的整体 tpf
    if summary["total_parallel_steps"] > 0:
        summary["overall_generated_tokens_including_eos_per_step"] = (
            summary["total_generated_tokens_including_eos"] / summary["total_parallel_steps"]
        )
        summary["overall_actual_tokens_excluding_eos_per_step"] = (
            summary["total_actual_tokens_excluding_eos"] / summary["total_parallel_steps"]
        )
    else:
        summary["overall_generated_tokens_including_eos_per_step"] = 0.0
        summary["overall_actual_tokens_excluding_eos_per_step"] = 0.0
    
    # 计算平均每个样例的时间 (time per sample)
    if summary["processed_samples"] > 0:
        summary["avg_time_per_sample"] = summary["total_time"] / summary["processed_samples"]
    else:
        summary["avg_time_per_sample"] = 0.0
    
    # 计算平均每个样例的生成token长度 (tokens per sample)
    if summary["processed_samples"] > 0:
        summary["avg_generated_tokens_including_eos_per_sample"] = (
            summary["total_generated_tokens_including_eos"] / summary["processed_samples"]
        )
        summary["avg_actual_tokens_excluding_eos_per_sample"] = (
            summary["total_actual_tokens_excluding_eos"] / summary["processed_samples"]
        )
    else:
        summary["avg_generated_tokens_including_eos_per_sample"] = 0.0
        summary["avg_actual_tokens_excluding_eos_per_sample"] = 0.0
    
    # 计算平均每个样例的步数 (steps per sample)
    if summary["processed_samples"] > 0:
        summary["avg_parallel_steps_per_sample"] = summary["total_parallel_steps"] / summary["processed_samples"]
    else:
        summary["avg_parallel_steps_per_sample"] = 0.0
    
    # 添加元数据
    summary["files_processed"] = num_ranks_found
    summary["files_missing"] = len(summary["rank_files_missing"])
    
    # 清理不需要在最终结果中显示的临时字段
    del summary["generated_tokens_including_eos_per_second_sum"]
    del summary["actual_tokens_excluding_eos_per_second_sum"]
    del summary["generated_tokens_including_eos_per_step_sum"]
    del summary["actual_tokens_excluding_eos_per_step_sum"]
    
    return summary


def save_summary(folder_path: str, summary_stats: Dict[str, Any]) -> str:
    """
    将汇总统计数据保存到json文件
    
    Args:
        folder_path: 目标文件夹路径
        summary_stats: 汇总统计数据
        
    Returns:
        保存的文件路径
    """
    output_file = os.path.join(folder_path, "summary_stats.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    return output_file


def process_folder(folder_path: str, verbose: bool = True) -> bool:
    """
    处理单个包含rank文件的文件夹
    
    Args:
        folder_path: 文件夹路径
        verbose: 是否打印详细信息
        
    Returns:
        处理是否成功
    """
    try:
        if verbose:
            print(f"Processing folder: {folder_path}")
        
        # 加载所有rank文件的数据
        total_stats = load_rank_stats(folder_path)
        
        # 计算汇总统计
        summary_stats = calculate_summary_stats(total_stats)
        
        # 保存结果
        output_file = save_summary(folder_path, summary_stats)
        
        if verbose:
            print(f"  Found {summary_stats['files_processed']}/8 rank files")
            print(f"  Total samples: {summary_stats['processed_samples']}")
            print(f"  Total generated tokens including EOS: {summary_stats['total_generated_tokens_including_eos']}")
            print(f"  Total actual tokens excluding EOS: {summary_stats['total_actual_tokens_excluding_eos']}")
            print(f"  Total parallel steps: {summary_stats['total_parallel_steps']}")
            print(f"  Total time: {summary_stats['total_time']:.2f}s")
            print(f"  Overall generated tps: {summary_stats['overall_generated_tokens_including_eos_per_second']:.4f}")
            print(f"  Overall actual tps: {summary_stats['overall_actual_tokens_excluding_eos_per_second']:.4f}")
            print(f"  Avg generated tps (per rank): {summary_stats['avg_generated_tokens_including_eos_per_second']:.4f}")
            print(f"  Avg actual tps (per rank): {summary_stats['avg_actual_tokens_excluding_eos_per_second']:.4f}")
            print(f"  Overall generated tpf: {summary_stats['overall_generated_tokens_including_eos_per_step']:.4f}")
            print(f"  Overall actual tpf: {summary_stats['overall_actual_tokens_excluding_eos_per_step']:.4f}")
            print(f"  Avg generated tpf (per rank): {summary_stats['avg_generated_tokens_including_eos_per_step']:.4f}")
            print(f"  Avg actual tpf (per rank): {summary_stats['avg_actual_tokens_excluding_eos_per_step']:.4f}")
            print(f"  Avg time/sample: {summary_stats['avg_time_per_sample']:.4f}s")
            print(f"  Avg generated tokens/sample: {summary_stats['avg_generated_tokens_including_eos_per_sample']:.2f}")
            print(f"  Avg actual tokens/sample: {summary_stats['avg_actual_tokens_excluding_eos_per_sample']:.2f}")
            print(f"  Avg parallel steps/sample: {summary_stats['avg_parallel_steps_per_sample']:.2f}")
            print(f"  Summary saved to: {output_file}")
            if summary_stats['files_missing']:
                print(f"  Missing files: {summary_stats['files_missing']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="汇总 rank 统计文件并计算两种 token 口径下的 tps/tpf")
    parser.add_argument("--root_path", nargs='?', default="/mnt/rl/xinyi/LoPA/scale_llada_d2f/results", 
                       help="要搜索的根目录路径 (默认为当前目录)")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="显示详细输出")
    parser.add_argument("--dry-run", action="store_true",
                       help="只显示找到的文件夹，不进行处理")
    
    args = parser.parse_args()
    
    root_path = os.path.abspath(args.root_path)
    
    if not os.path.exists(root_path):
        print(f"Error: Path {root_path} does not exist!")
        return 1
    
    print(f"Searching for rank stats folders in: {root_path}")
    print()
    
    # 找到所有包含rank_0_final_stats.json的文件夹
    rank_folders = find_rank_stats_folders(root_path)
    
    if not rank_folders:
        print("No folders containing rank_0 stats files found!")
        return 1
    
    print(f"Found {len(rank_folders)} folders with rank stats:")
    for folder in sorted(rank_folders):
        print(f"  {folder}")
    print()
    
    if args.dry_run:
        print("Dry run mode - no files will be processed.")
        return 0
    
    # 处理每个文件夹
    success_count = 0
    for folder in sorted(rank_folders):
        if process_folder(folder, args.verbose):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(rank_folders)} folders.")
    
    return 0 if success_count == len(rank_folders) else 1


if __name__ == "__main__":
    exit(main())
