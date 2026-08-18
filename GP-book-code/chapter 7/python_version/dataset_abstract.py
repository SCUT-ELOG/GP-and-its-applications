# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 7 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
数据集抽象基类
从C++原项目转换而来
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from typing import Any
from gep import GEPClassifier


class DatasetAbstract(ABC):
    """数据集抽象基类"""
    
    def __init__(self, train_file: str, class_num: int, line_num: int, attribute_num: int):
        """
        初始化数据集
        
        Args:
            train_file: 训练数据文件路径
            class_num: 分类类别数
            line_num: 训练样本数
            attribute_num: 特征数
        """
        self.INPUT_TRAIN_NAME = train_file
        self.INPUT_CLASS_NUM = class_num
        self.INPUT_LINE_NUM = line_num
        self.INPUT_ATTRIBUTE_NUM = attribute_num
        
        self.input_attrs = None  # 训练数据属性
        self.input_class = None  # 训练数据标签
    
    @abstractmethod
    def load_train_data(self):
        """加载训练数据（子类实现）"""
        pass
    
    @abstractmethod
    def load_test_data(self):
        """加载测试数据（子类实现）"""
        pass
    
    def train_and_test(self, classifier: Any, classifier_name: str):
        """
        使用sklearn分类器训练和测试
        
        Args:
            classifier: sklearn分类器对象
            classifier_name: 分类器名称
        """
        print(f"\n{'='*50}")
        print(f"{classifier_name}")
        print('='*50)
        
        # 训练
        classifier.fit(self.input_attrs, self.input_class)
        
        # 加载测试数据
        test_attrs, test_class = self.load_test_data()
        
        # 预测
        predictions = classifier.predict(test_attrs)
        
        # 计算准确率
        accuracy = accuracy_score(test_class, predictions)
        
        # 计算混淆矩阵
        cm = confusion_matrix(test_class, predictions)
        
        # 输出结果
        print(f"Test result: {np.sum(predictions == test_class)} right, "
              f"{np.sum(predictions != test_class)} wrong.")
        print("\nConfusion Matrix:")
        print(cm)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # 如果是二分类，计算召回率
        # 注意：匹配C++版本，计算类别0的召回率（pos_label=0）
        # C++代码: recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        if self.INPUT_CLASS_NUM == 2:
            recall = recall_score(test_class, predictions, pos_label=0, average='binary', zero_division=0)
            print(f"Recall: {recall:.4f}")
    
    def train_gep_and_test(self):
        """使用GEP算法训练和测试"""
        print(f"\n{'='*50}")
        print("Genetic Expression Programming (GEP)")
        print('='*50)
        
        # 统计各类别正例数
        positive_examples_num = [0] * self.INPUT_CLASS_NUM
        for class_label in self.input_class:
            positive_examples_num[int(class_label)] += 1
        
        print(f"Positive examples per class: {positive_examples_num}")
        
        # 创建GEP分类器
        gep = GEPClassifier()
        
        # 为每个类别训练规则
        for class_num in range(self.INPUT_CLASS_NUM):
            print(f"\n{'*'*40}")
            print(f"Training for class {class_num}")
            print('*'*40)
            gep.train_one_class(
                class_num,
                self.input_attrs,
                self.input_class,
                self.INPUT_LINE_NUM,
                self.INPUT_ATTRIBUTE_NUM,
                positive_examples_num
            )
        
        # 后剪枝
        print(f"\n{'*'*40}")
        print("Post-pruning")
        print('*'*40)
        gep.postpruning(
            self.input_class,
            self.INPUT_LINE_NUM,
            self.INPUT_CLASS_NUM,
            positive_examples_num
        )
        
        # 测试
        print(f"\n{'*'*40}")
        print("Testing")
        print('*'*40)
        test_attrs, test_class = self.load_test_data()
        input_test_num = len(test_class)
        
        predictions = gep.predict(test_attrs, input_test_num)
        
        # 计算统计信息
        unclassified = np.sum(predictions == gep.default_class)
        unclassified_right = np.sum((predictions == gep.default_class) & (predictions == test_class))
        unclassified_wrong = unclassified - unclassified_right
        
        # 计算混淆矩阵
        cm = confusion_matrix(test_class, predictions)
        
        # 输出结果
        print("\nTest result: Confusion matrix")
        print(cm)
        
        print(f"\nUnclassified: {unclassified} default, "
              f"{unclassified_right} right, {unclassified_wrong} wrong.")
        
        # 计算准确率
        accuracy = accuracy_score(test_class, predictions)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # 如果是二分类，计算召回率
        # 注意：匹配C++版本，计算类别0的召回率（pos_label=0）
        # C++代码: recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        if self.INPUT_CLASS_NUM == 2:
            recall = recall_score(test_class, predictions, pos_label=0, average='binary', zero_division=0)
            print(f"Recall: {recall:.4f}")
