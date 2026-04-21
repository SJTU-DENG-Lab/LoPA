#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理所有包含 samples_humaneval 的 jsonl 文件的脚本
在每个文件上运行 postprocess_code.py 并将输出保存到相同目录
"""

import os
import subprocess
import sys
from pathlib import Path

# os.environ["https_proxy"] = "http://proxy-node:7890"
# os.environ["http_proxy"] = "http://proxy-node:7890"
# os.environ["all_proxy"] = "socks5://proxy-node:7890"

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def find_humaneval_files(root_dir):
    """
    在指定目录下递归查找所有包含 'samples_humaneval' 的 jsonl 文件
    """
    humaneval_files = []
    
    # 使用 Path.rglob 递归搜索
    root_path = Path(root_dir)
    
    for file_path in root_path.rglob("*samples_humaneval*.jsonl"):
        humaneval_files.append(str(file_path))
    
    return humaneval_files

def process_file(jsonl_file, postprocess_script):
    """
    对单个 jsonl 文件运行 postprocess_code.py 并保存输出
    """
    try:
        print(f"正在处理文件: {jsonl_file}")
        
        # 运行 postprocess_code.py
        result = subprocess.run(
            ["python", postprocess_script, jsonl_file],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(jsonl_file)  # 在文件所在目录运行
        )
        
        # 获取输出文件路径（与jsonl文件在同一目录）
        output_file = os.path.join(os.path.dirname(jsonl_file), 
                                 f"{os.path.basename(jsonl_file)}_output.txt")
        
        # 保存标准输出到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 处理文件: {jsonl_file} ===\n")
            f.write(f"命令: python {postprocess_script} {jsonl_file}\n")
            f.write(f"返回码: {result.returncode}\n\n")
            f.write("=== 标准输出 ===\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== 标准错误 ===\n")
                f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"  ✓ 成功处理，输出保存到: {output_file}")
        else:
            print(f"  ✗ 处理失败 (返回码: {result.returncode})，详细信息保存到: {output_file}")
            
        return result.returncode == 0
        
    except Exception as e:
        error_msg = f"处理文件 {jsonl_file} 时发生错误: {str(e)}"
        print(f"  ✗ {error_msg}")
        
        # 保存错误信息
        error_file = os.path.join(os.path.dirname(jsonl_file), 
                                f"{os.path.basename(jsonl_file)}_error.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(error_msg)
        
        return False

def main():
    # 配置路径
    root_dir = "/mnt/rl/xinyi/LoPA/baselines/klass_my/src/evals_dream_klassno"
    postprocess_script = "/mnt/rl/xinyi/LoPA/scale_llada_d2f/postprocess_code.py"
    
    print(f"搜索目录: {root_dir}")
    print(f"后处理脚本: {postprocess_script}")
    print("=" * 60)
    
    # 检查后处理脚本是否存在
    if not os.path.exists(postprocess_script):
        print(f"错误: 后处理脚本不存在: {postprocess_script}")
        sys.exit(1)
    
    # 查找所有 humaneval 文件
    print("正在搜索包含 'samples_humaneval' 的 jsonl 文件...")
    humaneval_files = find_humaneval_files(root_dir)
    
    if not humaneval_files:
        print("未找到任何包含 'samples_humaneval' 的 jsonl 文件")
        return
    
    print(f"找到 {len(humaneval_files)} 个文件:")
    for i, file_path in enumerate(humaneval_files, 1):
        print(f"  {i}. {file_path}")
    
    print("\n" + "=" * 60)
    
    # 处理每个文件
    success_count = 0
    for i, jsonl_file in enumerate(humaneval_files, 1):
        print(f"\n[{i}/{len(humaneval_files)}] ", end="")
        if process_file(jsonl_file, postprocess_script):
            success_count += 1
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"总文件数: {len(humaneval_files)}")
    print(f"成功处理: {success_count}")
    print(f"失败处理: {len(humaneval_files) - success_count}")

if __name__ == "__main__":
    main() 