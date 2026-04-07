#!/usr/bin/env python3
"""
遍历文件夹，找到包含rank_0_final_stats.json的文件夹，
汇总rank_0到rank_7的统计数据，计算整体统计信息并保存到新的json文件中。
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def find_rank_stats_folders(root_path: str) -> List[str]:
    """
    遍历文件夹，找到所有包含rank_0_final_stats.json的文件夹
    
    Args:
        root_path: 根目录路径
        
    Returns:
        包含rank_0_final_stats.json的文件夹路径列表
    """
    rank_folders = []
    
    for root, dirs, files in os.walk(root_path):
        if "rank_0_final_stats.json" in files:
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
        "total_tokens_generated": 0,  # 新字段
        "total_steps_taken": 0,       # 新字段
        "total_time": 0.0,
        "tokens_per_second_sum": 0.0,  # 用于计算平均值
        "tokens_per_step_sum": 0.0,    # 用于计算平均值
        "rank_files_found": [],
        "rank_files_missing": []
    }
    
    # 检查rank_0到rank_7的文件
    for rank in range(8):
        rank_file = os.path.join(folder_path, f"rank_{rank}_final_stats.json")
        
        if os.path.exists(rank_file):
            try:
                with open(rank_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 累加数据 - 适配新的字段名
                total_stats["processed_samples"] += data.get("processed_samples", 0)
                total_stats["total_samples"] += data.get("total_samples", 0)
                total_stats["total_tokens_generated"] += data.get("total_tokens_generated (best paths)", 0)
                total_stats["total_steps_taken"] += data.get("total_steps_taken (best paths)", 0)
                total_stats["total_time"] += data.get("total_time", 0.0)
                
                # 累加每个rank的平均值，后续计算整体平均值
                total_stats["tokens_per_second_sum"] += data.get("tokens_per_second", 0.0)
                total_stats["tokens_per_step_sum"] += data.get("tokens_per_step (best paths avg)", 0.0)
                
                total_stats["rank_files_found"].append(f"rank_{rank}_final_stats.json")
                
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Error reading {rank_file}: {e}")
                total_stats["rank_files_missing"].append(f"rank_{rank}_final_stats.json")
        else:
            total_stats["rank_files_missing"].append(f"rank_{rank}_final_stats.json")
    
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
    
    # 计算整体平均tokens per second (基于总时间和总tokens)
    if summary["total_time"] > 0:
        summary["overall_tokens_per_second"] = summary["total_tokens_generated"] / summary["total_time"]
    else:
        summary["overall_tokens_per_second"] = 0.0
    
    # 计算各rank的平均tokens per second
    if num_ranks_found > 0:
        summary["avg_tokens_per_second"] = summary["tokens_per_second_sum"] / num_ranks_found
    else:
        summary["avg_tokens_per_second"] = 0.0
    
    # 计算各rank的平均tokens per step
    if num_ranks_found > 0:
        summary["avg_tokens_per_step"] = summary["tokens_per_step_sum"] / num_ranks_found
    else:
        summary["avg_tokens_per_step"] = 0.0
    
    # 计算整体平均tokens per step (基于总tokens和总steps)
    if summary["total_steps_taken"] > 0:
        summary["overall_tokens_per_step"] = summary["total_tokens_generated"] / summary["total_steps_taken"]
    else:
        summary["overall_tokens_per_step"] = 0.0
    
    # 计算平均每个样例的时间 (time per sample)
    if summary["processed_samples"] > 0:
        summary["avg_time_per_sample"] = summary["total_time"] / summary["processed_samples"]
    else:
        summary["avg_time_per_sample"] = 0.0
    
    # 计算平均每个样例的生成token长度 (tokens per sample)
    if summary["processed_samples"] > 0:
        summary["avg_tokens_per_sample"] = summary["total_tokens_generated"] / summary["processed_samples"]
    else:
        summary["avg_tokens_per_sample"] = 0.0
    
    # 计算平均每个样例的步数 (steps per sample)
    if summary["processed_samples"] > 0:
        summary["avg_steps_per_sample"] = summary["total_steps_taken"] / summary["processed_samples"]
    else:
        summary["avg_steps_per_sample"] = 0.0
    
    # 添加元数据
    summary["files_processed"] = num_ranks_found
    summary["files_missing"] = len(summary["rank_files_missing"])
    
    # 清理不需要在最终结果中显示的临时字段
    del summary["tokens_per_second_sum"]
    del summary["tokens_per_step_sum"]
    
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
            print(f"  Total tokens generated: {summary_stats['total_tokens_generated']}")
            print(f"  Total steps taken: {summary_stats['total_steps_taken']}")
            print(f"  Total time: {summary_stats['total_time']:.2f}s")
            print(f"  Overall tokens/sec: {summary_stats['overall_tokens_per_second']:.4f}")
            print(f"  Avg tokens/sec (per rank): {summary_stats['avg_tokens_per_second']:.4f}")
            print(f"  Overall tokens/step: {summary_stats['overall_tokens_per_step']:.4f}")
            print(f"  Avg tokens/step (per rank): {summary_stats['avg_tokens_per_step']:.4f}")
            print(f"  Avg time/sample: {summary_stats['avg_time_per_sample']:.4f}s")
            print(f"  Avg tokens/sample: {summary_stats['avg_tokens_per_sample']:.2f}")
            print(f"  Avg steps/sample: {summary_stats['avg_steps_per_sample']:.2f}")
            print(f"  Summary saved to: {output_file}")
            if summary_stats['files_missing']:
                print(f"  Missing files: {summary_stats['files_missing']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="汇总rank统计文件并计算整体统计指标")
    parser.add_argument("--root_path", nargs='?', default="/home/chenkai/data/D2F_2_xck/eval_dream_jointlog_analyse_rebuttal_sum", 
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
        print("No folders containing rank_0_final_stats.json found!")
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
