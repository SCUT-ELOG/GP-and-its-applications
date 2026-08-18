# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 7 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
简单测试脚本 - 运行MONK-1数据集
"""

import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.monk_dataset import Monk1

print("="*70)
print("GEP CLASSIFICATION - Simple Test")
print("="*70)

# 创建MONK-1数据集
print("\nLoading MONK-1 dataset...")
dataset = Monk1()

print(f"\nDataset: {dataset.__class__.__name__}")
print(f"Training samples: {dataset.INPUT_LINE_NUM}")
print(f"Features: {dataset.INPUT_ATTRIBUTE_NUM}")
print(f"Classes: {dataset.INPUT_CLASS_NUM}")

# 只运行GEP算法
print("\n" + "="*70)
print("Running GEP Algorithm")
print("="*70)

dataset.train_gep_and_test()

print("\n" + "="*70)
print("TEST COMPLETED")
print("="*70)
