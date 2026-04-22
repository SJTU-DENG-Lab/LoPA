#!/usr/bin/env python3
"""
遍历文件夹，找到包含 rank_0_final_stats.json 的文件夹，
汇总多卡运行的统计数据，计算整体统计信息（包含TPF）并保存到新的json文件中。
"""

import os
import json
import argparse
from typing import Dict, List, Any


def find_rank_stats_folders(root_path: str) -> List[str]:
    """
    遍历文件夹，找到所有包含 rank_0_final_stats.json 的文件夹
    
    Args:
        root_path: 根目录路径
        
    Returns:
        包含 rank_0_final_stats.json 的文件夹路径列表
    """
    rank_folders = []
    
    for root, dirs, files in os.walk(root_path):
        # [修改] 匹配生成代码一保留的文件名
        if "rank_0_final_stats.json" in files:
            rank_folders.append(root)
    
    return rank_folders


def load_rank_stats(folder_path: str, max_ranks: int = 8) -> Dict[str, Any]:
    """
    加载一个文件夹中所有rank文件的统计数据
    
    Args:
        folder_path: 包含rank文件的文件夹路径
        max_ranks: 预期最大的rank数量 (默认8)
        
    Returns:
        汇总的统计数据
    """
    total_stats = {
        "total_non_eos_tokens": 0,
        "total_nfe": 0,
        "total_time": 0.0,
        "rank_files_found": [],
        "rank_files_missing": []
    }
    
    for rank in range(max_ranks):
        # [修改] 匹配生成代码一保留的文件名格式
        rank_file_name = f"rank_{rank}_final_stats.json"
        rank_file = os.path.join(folder_path, rank_file_name)
        
        if os.path.exists(rank_file):
            try:
                with open(rank_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # [修改] 映射生成代码一保留的 JSON 键值
                total_stats["total_non_eos_tokens"] += data.get("total_tokens", 0)
                total_stats["total_nfe"] += data.get("total_used_steps", 0)
                total_stats["total_time"] += data.get("total_time", 0.0) 
                total_stats["rank_files_found"].append(rank_file_name)
                
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Error reading {rank_file}: {e}")
                total_stats["rank_files_missing"].append(rank_file_name)
        else:
            total_stats["rank_files_missing"].append(rank_file_name)
    
    return total_stats


def calculate_summary_stats(total_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据汇总数据计算统计指标，特别是 TPF
    
    Args:
        total_stats: 汇总的统计数据
        
    Returns:
        包含计算指标的统计数据
    """
    summary = total_stats.copy()
    
    # 核心指标 1：TPF (Tokens Per Forward - 每一步平均解码的token数量)
    if summary["total_nfe"] > 0:
        summary["TPF_avg_tokens_per_forward"] = summary["total_non_eos_tokens"] / summary["total_nfe"]
    else:
        summary["TPF_avg_tokens_per_forward"] = 0.0

    # 核心指标 2：平均每个 token 消耗的 NFE（TPF的倒数）
    if summary["total_non_eos_tokens"] > 0:
        summary["avg_nfe_per_token"] = summary["total_nfe"] / summary["total_non_eos_tokens"]
    else:
        summary["avg_nfe_per_token"] = 0.0

    # 核心指标 3：总系统吞吐量 (Tokens per second)
    # 注意：这里的 total_time 是累加的。如果各 rank 是绝对并行运行的，实际墙钟时间应该是各 rank 的最大值。
    # 这里保持和你原代码一致，或者你可以选择除以 (total_time / ranks_found) 获取平均吞吐
    ranks_count = len(summary["rank_files_found"])
    if summary["total_time"] > 0 and ranks_count > 0:
        # 估算实际经历的墙钟时间 (所有节点的平均耗时)
        avg_wall_clock_time = summary["total_time"] / ranks_count
        summary["avg_tokens_per_second_system_wide"] = summary["total_non_eos_tokens"] / avg_wall_clock_time
    else:
        summary["avg_tokens_per_second_system_wide"] = 0.0
    
    # 添加元数据
    summary["files_processed"] = len(summary["rank_files_found"])
    summary["files_missing"] = len(summary["rank_files_missing"])
    
    return summary


def save_summary(folder_path: str, summary_stats: Dict[str, Any]) -> str:
    """
    将汇总统计数据保存到json文件
    """
    output_file = os.path.join(folder_path, "summary_tpf_stats.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=4, ensure_ascii=False)
    
    return output_file


def process_folder(folder_path: str, verbose: bool = True) -> bool:
    """
    处理单个包含rank文件的文件夹
    """
    try:
        if verbose:
            print(f"\nProcessing folder: {folder_path}")
        
        # 加载所有rank文件的数据
        total_stats = load_rank_stats(folder_path)
        
        # 如果没有找到任何文件，跳过
        if not total_stats["rank_files_found"]:
            print(f"  No valid rank files found in {folder_path}")
            return False

        # 计算汇总统计
        summary_stats = calculate_summary_stats(total_stats)
        
        # 保存结果
        output_file = save_summary(folder_path, summary_stats)
        
        if verbose:
            print(f"  Found {summary_stats['files_processed']} rank files")
            print(f"  Total non-EOS tokens: {summary_stats['total_non_eos_tokens']}")
            print(f"  Total NFE (forward passes): {summary_stats['total_nfe']}")
            print(f"  ------------------------------------------------")
            print(f"  > TPF (Tokens Per Forward): {summary_stats['TPF_avg_tokens_per_forward']:.4f} tokens/step")
            print(f"  > Avg NFE per token: {summary_stats['avg_nfe_per_token']:.4f} steps/token")
            print(f"  > System-wide Throughput: {summary_stats['avg_tokens_per_second_system_wide']:.2f} tokens/sec")
            print(f"  ------------------------------------------------")
            print(f"  Summary saved to: {output_file}")
            
            if summary_stats['files_missing'] > 0:
                print(f"  Warning: Missing {summary_stats['files_missing']} files (e.g., {summary_stats['rank_files_missing'][:2]}...)")
        
        return True
        
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="汇总基于 DiffLLM 生成的统计文件并计算整体 TPF")
    # 默认路径可以改成你运行 lm-eval 时的保存路径，比如 "generation_stats"
    parser.add_argument("root_path", nargs='?', default="/home/chenkai/data/LoPA/baselines/dParallel-master/Dream/eval_instruct/evals_dream_dp_my_new", 
                        help="要搜索的根目录路径")
    parser.add_argument("-v", "--verbose", action="store_true", default=True,
                        help="显示详细输出")
    parser.add_argument("--dry-run", action="store_true",
                        help="只显示找到的文件夹，不进行处理")
    
    args = parser.parse_args()
    root_path = os.path.abspath(args.root_path)
    
    if not os.path.exists(root_path):
        print(f"Error: Path {root_path} does not exist!")
        return 1
    
    print(f"Searching for rank stats folders in: {root_path}")
    
    # 找到所有包含 rank_0_final_stats.json 的文件夹
    rank_folders = find_rank_stats_folders(root_path)
    
    if not rank_folders:
        print("No folders containing rank_0_final_stats.json found!")
        return 1
    
    print(f"Found {len(rank_folders)} folders with rank stats.")
    
    if args.dry_run:
        print("Dry run mode - no files will be processed.")
        return 0
    
    success_count = 0
    for folder in sorted(rank_folders):
        if process_folder(folder, args.verbose):
            success_count += 1
            
    print(f"\nSuccessfully processed {success_count}/{len(rank_folders)} folders.")
    return 0 if success_count == len(rank_folders) else 1


if __name__ == "__main__":
    exit(main())