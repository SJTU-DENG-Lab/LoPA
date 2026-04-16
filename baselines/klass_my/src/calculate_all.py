#!/usr/bin/env python3
"""
遍历文件夹，找到包含 rank_0_final_stats.json 的文件夹，
汇总多卡运行的统计数据，计算两种情况的整体统计信息（包含TPF）并保存到新的json文件中。
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
        "total_tokens_with_eos": 0,
        "total_tokens_without_eos": 0,
        "total_nfe": 0,
        "total_time": 0.0,
        "rank_files_found": [],
        "rank_files_missing": []
    }
    
    for rank in range(max_ranks):
        rank_file_name = f"rank_{rank}_final_stats.json"
        rank_file = os.path.join(folder_path, rank_file_name)
        
        if os.path.exists(rank_file):
            try:
                with open(rank_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 读取新代码保存的两个 Token 指标
                total_stats["total_tokens_with_eos"] += data.get("total_tokens_with_eos", 0)
                total_stats["total_tokens_without_eos"] += data.get("total_tokens_without_eos", 0)
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
    根据汇总数据计算统计指标，分别计算含EOS和不含EOS的TPF
    
    Args:
        total_stats: 汇总的统计数据
        
    Returns:
        包含计算指标的统计数据
    """
    summary = total_stats.copy()
    
    # 核心指标 1：TPF (Tokens Per Forward - 每一步平均解码的token数量)
    if summary["total_nfe"] > 0:
        summary["TPF_with_eos"] = summary["total_tokens_with_eos"] / summary["total_nfe"]
        summary["TPF_without_eos"] = summary["total_tokens_without_eos"] / summary["total_nfe"]
    else:
        summary["TPF_with_eos"] = 0.0
        summary["TPF_without_eos"] = 0.0

    # 核心指标 2：平均每个 token 消耗的 NFE（TPF的倒数）
    if summary["total_tokens_with_eos"] > 0:
        summary["avg_nfe_per_token_with_eos"] = summary["total_nfe"] / summary["total_tokens_with_eos"]
    else:
        summary["avg_nfe_per_token_with_eos"] = 0.0

    if summary["total_tokens_without_eos"] > 0:
        summary["avg_nfe_per_token_without_eos"] = summary["total_nfe"] / summary["total_tokens_without_eos"]
    else:
        summary["avg_nfe_per_token_without_eos"] = 0.0

    # 核心指标 3：总系统吞吐量 (Tokens per second)
    ranks_count = len(summary["rank_files_found"])
    if summary["total_time"] > 0 and ranks_count > 0:
        # 估算实际经历的墙钟时间 (所有节点的平均耗时)
        avg_wall_clock_time = summary["total_time"] / ranks_count
        summary["system_throughput_with_eos"] = summary["total_tokens_with_eos"] / avg_wall_clock_time
        summary["system_throughput_without_eos"] = summary["total_tokens_without_eos"] / avg_wall_clock_time
    else:
        summary["system_throughput_with_eos"] = 0.0
        summary["system_throughput_without_eos"] = 0.0
    
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
            print(f"  Total NFE (forward passes): {summary_stats['total_nfe']}")
            print(f"  --- [包含 EOS/Padding 的统计] ---")
            print(f"  > Total tokens (with EOS): {summary_stats['total_tokens_with_eos']}")
            print(f"  > TPF: {summary_stats['TPF_with_eos']:.4f} tokens/step")
            print(f"  > Avg NFE per token: {summary_stats['avg_nfe_per_token_with_eos']:.4f} steps/token")
            print(f"  > System Throughput: {summary_stats['system_throughput_with_eos']:.2f} tokens/sec")
            print(f"  --- [截断至 EOS 前的有效统计] ---")
            print(f"  > Total tokens (without EOS): {summary_stats['total_tokens_without_eos']}")
            print(f"  > TPF: {summary_stats['TPF_without_eos']:.4f} tokens/step")
            print(f"  > Avg NFE per token: {summary_stats['avg_nfe_per_token_without_eos']:.4f} steps/token")
            print(f"  > System Throughput: {summary_stats['system_throughput_without_eos']:.2f} tokens/sec")
            print(f"  ------------------------------------------------")
            print(f"  Summary saved to: {output_file}")
            
            if summary_stats['files_missing'] > 0:
                print(f"  Warning: Missing {summary_stats['files_missing']} files (e.g., {summary_stats['rank_files_missing'][:2]}...)")
        
        return True
        
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="汇总基于 DiffLLM 生成的统计文件并计算两种整体 TPF")
    # 默认路径可以改成你运行 lm-eval 时的保存路径
    parser.add_argument("root_path", nargs='?', default="/mnt/rl/xinyi/klass_my/src/evals_dream_seed1234_noeos", 
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