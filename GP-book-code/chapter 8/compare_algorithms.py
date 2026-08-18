# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 8 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    # 如果字体不可用，使用系统默认字体
    pass

def load_pggp_data(file_path):
    """加载PGGP算法的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fitness_data = []
    for experiment in data['all_experiments']:
        fitness_data.append(experiment['fitness_trend'])
    
    return fitness_data

def load_stgp_data(file_path):
    """加载stGP算法的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fitness_data = []
    for result in data['all_results']:
        fitness_data.append(result['logbook']['best_fitness'])
    
    return fitness_data

def calculate_statistics(fitness_data):
    """计算每一轮的统计数据：中位数、0.25分位数、0.75分位数"""
    # 转换为numpy数组，处理不同长度的序列
    max_length = max(len(run) for run in fitness_data)
    
    # 用NaN填充较短的序列
    padded_data = []
    for run in fitness_data:
        padded_run = run + [np.nan] * (max_length - len(run))
        padded_data.append(padded_run)
    
    fitness_array = np.array(padded_data)
    
    # 计算统计数据，忽略NaN值
    medians = np.nanmedian(fitness_array, axis=0)
    q25 = np.nanpercentile(fitness_array, 25, axis=0)
    q75 = np.nanpercentile(fitness_array, 75, axis=0)
    
    # 移除末尾的NaN值
    valid_indices = ~np.isnan(medians)
    
    return medians[valid_indices], q25[valid_indices], q75[valid_indices]

def plot_comparison(pggp_stats, stgp_stats, dataset_name, output_dir):
    """绘制两个算法的比较图"""
    pggp_median, pggp_q25, pggp_q75 = pggp_stats
    stgp_median, stgp_q25, stgp_q75 = stgp_stats
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 生成x轴（代数）
    pggp_generations = range(1, len(pggp_median) + 1)
    stgp_generations = range(1, len(stgp_median) + 1)
    
    # 绘制PGGP算法的结果
    plt.fill_between(pggp_generations, pggp_q25, pggp_q75, 
                     alpha=0.1, color='blue', label='PGGP Quartile Range')
    plt.plot(pggp_generations, pggp_median, 'b-', linewidth=1, 
             label='PGGP Median', marker='o', markersize=1)
    
    # 绘制stGP算法的结果
    plt.fill_between(stgp_generations, stgp_q25, stgp_q75, 
                     alpha=0.1, color='red', label='stGP Quartile Range')
    plt.plot(stgp_generations, stgp_median, 'r-', linewidth=1, 
             label='stGP Median', marker='s', markersize=1)
    
    # 设置图形属性
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    # plt.title(f'Algorithm Comparison - Dataset {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴
    plt.xlim(1, max(len(pggp_median), len(stgp_median)))
    
    # 保存图片 - 使用矢量图格式
    output_path_svg = os.path.join(output_dir, f'comparison_{dataset_name}.svg')
    output_path_pdf = os.path.join(output_dir, f'comparison_{dataset_name}.pdf')
    plt.tight_layout()
    
    # 保存为SVG和PDF两种矢量图格式
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"已生成比较图: {output_path_svg} 和 {output_path_pdf}")

def main():
    """主函数"""
    # 设置路径
    results_dir = Path("results")
    output_dir = Path("comparison_plots")
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有JSON文件
    all_files = list(results_dir.glob("*.json"))
    
    # 按算法前缀分组
    pggp_files = [f for f in all_files if f.name.startswith("pggp_")]
    stgp_files = [f for f in all_files if f.name.startswith("stGP_")]
    
    # 提取数据集名称（去掉前缀）
    pggp_datasets = {f.name.replace("pggp_", "").replace(".json", ""): f for f in pggp_files}
    stgp_datasets = {f.name.replace("stGP_", "").replace(".json", ""): f for f in stgp_files}
    
    # 找到对应的数据集
    common_datasets = set(pggp_datasets.keys()).intersection(set(stgp_datasets.keys()))
    
    print(f"找到 {len(common_datasets)} 组对应的数据文件")
    
    # 为每组对应的数据集生成比较图
    for dataset_name in sorted(common_datasets):
        print(f"\\n处理数据集: {dataset_name}")
        
        # 加载数据
        pggp_file = pggp_datasets[dataset_name]
        stgp_file = stgp_datasets[dataset_name]
        
        pggp_fitness = load_pggp_data(pggp_file)
        stgp_fitness = load_stgp_data(stgp_file)
        
        print(f"PGGP: {len(pggp_fitness)} 次运行")
        print(f"stGP: {len(stgp_fitness)} 次运行")
        
        # 计算统计数据
        pggp_stats = calculate_statistics(pggp_fitness)
        stgp_stats = calculate_statistics(stgp_fitness)
        
        print(f"PGGP: {len(pggp_stats[0])} 代")
        print(f"stGP: {len(stgp_stats[0])} 代")
        
        # 绘制比较图
        plot_comparison(pggp_stats, stgp_stats, dataset_name, output_dir)
    
    print(f"\\n所有比较图已生成完成，保存在 {output_dir} 目录中")

if __name__ == "__main__":
    main()