# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 7 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
主程序 - 算法对比
从C++原项目转换而来
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 导入数据集类
from datasets.monk_dataset import Monk1, Monk2, Monk3
from datasets.other_datasets import Iris, HabermanSurvival, Zoo


def run_traditional_algorithms(dataset):
    """运行传统机器学习算法（参数对齐C++版本）"""
    
    print("\n" + "="*70)
    print("TRADITIONAL MACHINE LEARNING ALGORITHMS")
    print("="*70)
    
    # 设置随机种子，匹配C++的 cv::theRNG().state = 1
    import random
    random.seed(1)
    np.random.seed(1)
    
    # 决策树 - 匹配C++: setMaxDepth(10), setCVFolds(0)
    # 使用random_state=1以匹配C++的全局随机种子
    dtree = DecisionTreeClassifier(max_depth=10, random_state=1)
    dataset.train_and_test(dtree, "Decision Tree")
    
    # 朴素贝叶斯 - 匹配C++: NormalBayesClassifier::create()
    # 高斯朴素贝叶斯，无需额外参数
    bayes = GaussianNB()
    dataset.train_and_test(bayes, "Naive Bayes")
    
    # 随机森林 - 匹配C++: setMaxCategories(), setUseSurrogates(false)
    # OpenCV RTrees默认参数: 树数量通常为50-100, max_depth无限制
    # 为了更接近C++行为，使用max_depth=None（无限制）
    rforest = RandomForestClassifier(
        n_estimators=100,      # sklearn默认100，OpenCV可能类似
        max_depth=None,        # 无限制，匹配C++默认行为
        random_state=1         # 匹配C++的全局随机种子=1
    )
    dataset.train_and_test(rforest, "Random Forest")
    
    # SVM - 匹配C++: C_SVC, RBF核, TermCriteria(EPS, 100000, 1e-8)
    # C++未明确设置gamma，OpenCV默认通常是1/n_features（即'auto'）
    svm = SVC(
        kernel='rbf',          # RBF核，匹配C++
        C=1.0,                 # 正则化参数，匹配OpenCV默认
        gamma='auto',          # 1/n_features，匹配OpenCV默认行为
        tol=1e-8,              # 终止准则精度，匹配C++的1e-8
        max_iter=100000,       # 最大迭代次数，匹配C++的100000
        random_state=1         # 匹配C++的全局随机种子=1
    )
    dataset.train_and_test(svm, "Support Vector Machine (SVM)")


def run_gep_algorithm(dataset):
    """运行GEP算法"""
    
    print("\n" + "="*70)
    print("GENE EXPRESSION PROGRAMMING (GEP)")
    print("="*70)
    
    dataset.train_gep_and_test()


def main():
    """主函数"""
    
    print("="*70)
    print("GEP CLASSIFICATION - Python Version")
    print("="*70)
    print("\nSelect a dataset:")
    print("1. MONK-1")
    print("2. MONK-2")
    print("3. MONK-3")
    print("4. Iris")
    print("5. Haberman Survival")
    print("6. Zoo")
    
    # 默认使用MONK-1数据集
    choice = input("\nEnter your choice (1-6, default is 1): ").strip()
    
    if choice == "2":
        dataset = Monk2()
    elif choice == "3":
        dataset = Monk3()
    elif choice == "4":
        dataset = Iris()
    elif choice == "5":
        dataset = HabermanSurvival()
    elif choice == "6":
        dataset = Zoo()
    else:
        dataset = Monk1()
    
    print(f"\nDataset: {dataset.__class__.__name__}")
    print(f"Training samples: {dataset.INPUT_LINE_NUM}")
    print(f"Features: {dataset.INPUT_ATTRIBUTE_NUM}")
    print(f"Classes: {dataset.INPUT_CLASS_NUM}")
    
    # 选择运行模式
    print("\nSelect mode:")
    print("1. Run all algorithms (Traditional + GEP)")
    print("2. Run traditional algorithms only")
    print("3. Run GEP only")
    
    mode = input("\nEnter your choice (1-3, default is 1): ").strip()
    
    if mode == "2":
        run_traditional_algorithms(dataset)
    elif mode == "3":
        run_gep_algorithm(dataset)
    else:
        run_traditional_algorithms(dataset)
        run_gep_algorithm(dataset)
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
