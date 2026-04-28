"""
DCA (Decision-aware Cross-modal Attention Masking) 实验脚本

运行DCA攻击并与标准MI-FGSM进行对比实验。
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
import subprocess
from datetime import datetime


def run_attack(attack_type, config_name="ensemble_3models", overrides=None):
    """
    运行指定类型的攻击
    
    Args:
        attack_type: 攻击类型 ('mifgsm' 或 'dca')
        config_name: 配置文件名
        overrides: 额外的配置覆盖
    
    Returns:
        str: 输出目录路径
    """
    cmd = [
        "python", "generate_adversarial_samples.py",
        f"--config-name={config_name}",
        f"attack={attack_type}",
    ]
    
    if overrides:
        cmd.extend(overrides)
    
    print(f"\n{'='*60}")
    print(f"Running {attack_type.upper()} attack...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Error running {attack_type} attack!")
        return None
    
    # 从输出中提取config hash（简化处理，实际应从日志解析）
    # 这里假设输出目录在 ./LAT/img/ 下
    output_base = "./LAT/img"
    if os.path.exists(output_base):
        dirs = sorted([d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))])
        if dirs:
            return os.path.join(output_base, dirs[-1])
    
    return None


def run_evaluation(adv_img_dir, model_name="doubao"):
    """
    运行黑盒评估
    
    Args:
        adv_img_dir: 对抗图像目录
        model_name: 黑盒模型名称
    
    Returns:
        dict: 评估结果
    """
    print(f"\n{'='*60}")
    print(f"Running blackbox evaluation with {model_name}...")
    print(f"{'='*60}\n")
    
    # 生成描述
    desc_cmd = [
        "python", "myblackbox_text_generation.py",
        f"blackbox.model_name={model_name}",
        f"data.adv_img_dir={adv_img_dir}",
    ]
    
    subprocess.run(desc_cmd, capture_output=False, text=True)
    
    # 语义相似度评估
    eval_cmd = [
        "python", "doubao_evaluate.py",
        f"blackbox.model_name={model_name}",
    ]
    
    subprocess.run(eval_cmd, capture_output=False, text=True)
    
    # 关键词匹配评估
    keyword_cmd = [
        "python", "keyword_matching_doubao.py",
        f"blackbox.model_name={model_name}",
    ]
    
    subprocess.run(keyword_cmd, capture_output=False, text=True)
    
    return {}


def compare_experiments(dca_dir, baseline_dir):
    """
    对比DCA和基线方法的实验结果
    
    Args:
        dca_dir: DCA攻击输出目录
        baseline_dir: 基线攻击输出目录
    
    Returns:
        dict: 对比结果
    """
    print(f"\n{'='*60}")
    print("Comparing DCA vs Baseline...")
    print(f"{'='*60}\n")
    
    comparison = {
        "dca_output": dca_dir,
        "baseline_output": baseline_dir,
        "timestamp": datetime.now().isoformat(),
    }
    
    # TODO: 从评估结果文件中提取具体指标进行对比
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Run DCA experiments")
    parser.add_argument(
        "--attack", 
        type=str, 
        default="dca",
        choices=["dca", "mifgsm", "both"],
        help="Attack type to run"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="ensemble_3models",
        help="Configuration name"
    )
    parser.add_argument(
        "--eval", 
        action="store_true",
        help="Run evaluation after attack"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="doubao",
        choices=["doubao", "qwen", "gpt4o"],
        help="Blackbox model for evaluation"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# DCA Experiment Runner")
    print(f"# Attack: {args.attack}")
    print(f"# Config: {args.config}")
    print(f"# Evaluation: {args.eval}")
    print(f"{'#'*60}\n")
    
    results = {}
    
    if args.attack in ["dca", "both"]:
        # 运行DCA攻击
        dca_overrides = [
            "dca.use_ggm=true",
            "dca.ggm_sigma=3.0",
        ]
        dca_dir = run_attack("dca", args.config, dca_overrides)
        results["dca"] = dca_dir
        
        if args.eval and dca_dir:
            run_evaluation(dca_dir, args.model)
    
    if args.attack in ["mifgsm", "both"]:
        # 运行基线MI-FGSM攻击
        baseline_dir = run_attack("mifgsm", args.config)
        results["baseline"] = baseline_dir
        
        if args.eval and baseline_dir:
            run_evaluation(baseline_dir, args.model)
    
    if args.attack == "both" and results.get("dca") and results.get("baseline"):
        # 对比实验
        comparison = compare_experiments(results["dca"], results["baseline"])
        
        # 保存对比结果
        output_file = f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nComparison results saved to {output_file}")
    
    print(f"\n{'#'*60}")
    print(f"# Experiment completed!")
    print(f"# Results: {results}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
