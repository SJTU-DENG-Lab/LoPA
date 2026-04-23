#!/usr/bin/env python3
"""
遍历文件夹，找到包含 rank 统计文件的文件夹，
汇总 rank_0 到 rank_7 的统计数据，计算整体统计信息并保存到新的 json 文件中。
兼容旧文件名格式、旧字段名以及新的 JSON 字段名。
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def find_rank_stats_folders(root_path: str) -> List[str]:
    """
    遍历文件夹，找到所有包含 rank_0 统计文件的文件夹
    """
    rank_folders = []
    
    for root, dirs, files in os.walk(root_path):
        # 兼容多种可能的 rank_0 文件名
        target_files = {"rank_0_final_stats.json", "rank_0_stats.json", "rank_0_final_stats.jsonl"}
        if any(f in files for f in target_files):
            rank_folders.append(root)
    
    return rank_folders


def load_rank_stats(folder_path: str) -> Dict[str, Any]:
    """
    加载一个文件夹中所有rank文件的统计数据，兼容新旧字段映射。
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
    
    for rank in range(8):
        rank_file = None
        candidate_files = [
            os.path.join(folder_path, f"rank_{rank}_stats.json"),
            os.path.join(folder_path, f"rank_{rank}_final_stats.json"),
        ]
        for candidate in candidate_files:
            if os.path.exists(candidate):
                rank_file = candidate
                break
        
        if rank_file is not None:
            try:
                with open(rank_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # --- 字段兼容性提取逻辑 ---
                # 1. 基础样本数与时间
                samples = data.get("processed_samples", 0)
                total_samples = data.get("total_samples", 0)
                time_val = data.get("total_time", 0.0)
                
                # 2. Token 总数兼容
                # 旧: total_generated_tokens_including_eos | 新: total_tokens_with_eos
                tokens_eos = data.get("total_generated_tokens_including_eos", data.get("total_tokens_with_eos", 0))
                # 旧: total_actual_tokens_excluding_eos | 新: total_tokens_without_eos
                tokens_no_eos = data.get("total_actual_tokens_excluding_eos", data.get("total_tokens_without_eos", 0))
                
                # 3. 步数兼容
                # 旧: total_parallel_steps | 新: total_used_steps
                steps = data.get("total_parallel_steps", data.get("total_used_steps", 0))
                
                # 4. TPS (Tokens Per Second) 兼容
                tps_eos = data.get("generated_tokens_including_eos_per_second", data.get("tokens_per_second_with_eos", 0.0))
                tps_no_eos = data.get("actual_tokens_excluding_eos_per_second", data.get("tokens_per_second_without_eos", 0.0))
                
                # 5. TPF (Tokens Per Step/Frame) 兼容处理
                # 如果是新格式，可能没有预计算好的 TPF，手动计算
                tpf_eos = data.get("generated_tokens_including_eos_per_step", 0.0)
                if tpf_eos == 0 and steps > 0:
                    tpf_eos = tokens_eos / steps
                    
                tpf_no_eos = data.get("actual_tokens_excluding_eos_per_step", 0.0)
                if tpf_no_eos == 0 and steps > 0:
                    tpf_no_eos = tokens_no_eos / steps

                # --- 累加数据 ---
                total_stats["processed_samples"] += samples
                total_stats["total_samples"] += total_samples
                total_stats["total_generated_tokens_including_eos"] += tokens_eos
                total_stats["total_actual_tokens_excluding_eos"] += tokens_no_eos
                total_stats["total_parallel_steps"] += steps
                total_stats["total_time"] += time_val
                
                total_stats["generated_tokens_including_eos_per_second_sum"] += tps_eos
                total_stats["actual_tokens_excluding_eos_per_second_sum"] += tps_no_eos
                total_stats["generated_tokens_including_eos_per_step_sum"] += tpf_eos
                total_stats["actual_tokens_excluding_eos_per_step_sum"] += tpf_no_eos
                
                total_stats["rank_files_found"].append(os.path.basename(rank_file))
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {rank_file}: {e}")
                total_stats["rank_files_missing"].append(f"rank_{rank}")
        else:
            total_stats["rank_files_missing"].append(f"rank_{rank}")
    
    return total_stats


def calculate_summary_stats(total_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据汇总数据计算统计指标
    """
    summary = total_stats.copy()
    num_ranks_found = len(summary["rank_files_found"])
    
    # 计算整体 tps (总 tokens / 总时间)
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
    
    # 计算跨 rank 平均 tps
    if num_ranks_found > 0:
        summary["avg_generated_tokens_including_eos_per_second"] = (
            summary["generated_tokens_including_eos_per_second_sum"] / num_ranks_found
        )
        summary["avg_actual_tokens_excluding_eos_per_second"] = (
            summary["actual_tokens_excluding_eos_per_second_sum"] / num_ranks_found
        )
        summary["avg_generated_tokens_including_eos_per_step"] = (
            summary["generated_tokens_including_eos_per_step_sum"] / num_ranks_found
        )
        summary["avg_actual_tokens_excluding_eos_per_step"] = (
            summary["actual_tokens_excluding_eos_per_step_sum"] / num_ranks_found
        )
    else:
        summary["avg_generated_tokens_including_eos_per_second"] = 0.0
        summary["avg_actual_tokens_excluding_eos_per_second"] = 0.0
        summary["avg_generated_tokens_including_eos_per_step"] = 0.0
        summary["avg_actual_tokens_excluding_eos_per_step"] = 0.0
    
    # 计算整体 tpf
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
    
    # 样平均指标
    if summary["processed_samples"] > 0:
        summary["avg_time_per_sample"] = summary["total_time"] / summary["processed_samples"]
        summary["avg_generated_tokens_including_eos_per_sample"] = (
            summary["total_generated_tokens_including_eos"] / summary["processed_samples"]
        )
        summary["avg_actual_tokens_excluding_eos_per_sample"] = (
            summary["total_actual_tokens_excluding_eos"] / summary["processed_samples"]
        )
        summary["avg_parallel_steps_per_sample"] = summary["total_parallel_steps"] / summary["processed_samples"]
    else:
        summary["avg_time_per_sample"] = 0.0
        summary["avg_generated_tokens_including_eos_per_sample"] = 0.0
        summary["avg_actual_tokens_excluding_eos_per_sample"] = 0.0
        summary["avg_parallel_steps_per_sample"] = 0.0
    
    summary["files_processed"] = num_ranks_found
    summary["files_missing_count"] = len(summary["rank_files_missing"])
    
    # 清理中间辅助字段
    fields_to_del = [
        "generated_tokens_including_eos_per_second_sum",
        "actual_tokens_excluding_eos_per_second_sum",
        "generated_tokens_including_eos_per_step_sum",
        "actual_tokens_excluding_eos_per_step_sum"
    ]
    for field in fields_to_del:
        if field in summary:
            del summary[field]
    
    return summary


def save_summary(folder_path: str, summary_stats: Dict[str, Any]) -> str:
    """
    将汇总统计数据保存到json文件
    """
    output_file = os.path.join(folder_path, "summary_stats.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    return output_file


def process_folder(folder_path: str, verbose: bool = True) -> bool:
    """
    处理单个包含rank文件的文件夹
    """
    try:
        if verbose:
            print(f"Processing folder: {folder_path}")
        
        total_stats = load_rank_stats(folder_path)
        summary_stats = calculate_summary_stats(total_stats)
        output_file = save_summary(folder_path, summary_stats)
        
        if verbose:
            print(f"  Found {summary_stats['files_processed']}/8 rank files")
            print(f"  Total samples: {summary_stats['processed_samples']}")
            print(f"  Overall actual tps: {summary_stats['overall_actual_tokens_excluding_eos_per_second']:.4f}")
            print(f"  Avg actual tps (per rank): {summary_stats['avg_actual_tokens_excluding_eos_per_second']:.4f}")
            print(f"  Summary saved to: {output_file}")
            if summary_stats['rank_files_missing']:
                print(f"  Missing ranks: {summary_stats['rank_files_missing']}")
            print()
        
        return True
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="汇总 rank 统计文件并计算两种 token 口径下的 tps/tpf (兼容新旧格式)")
    parser.add_argument("--root_path", nargs='?', default="/home/chenkai/data/LoPA", 
                        help="要搜索的根目录路径")
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
    rank_folders = find_rank_stats_folders(root_path)
    
    if not rank_folders:
        print("No folders containing rank stats files found!")
        return 1
    
    print(f"Found {len(rank_folders)} folders with rank stats.\n")
    
    if args.dry_run:
        for folder in sorted(rank_folders):
            print(f"  [Dry Run] Found: {folder}")
        return 0
    
    success_count = 0
    for folder in sorted(rank_folders):
        if process_folder(folder, args.verbose):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(rank_folders)} folders.")
    return 0 if success_count == len(rank_folders) else 1


if __name__ == "__main__":
    exit(main())