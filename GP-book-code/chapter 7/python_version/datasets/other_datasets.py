# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 7 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
Iris和Haberman数据集实现
从C++原项目转换而来
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_abstract import DatasetAbstract


class Iris(DatasetAbstract):
    """Iris（鸢尾花）数据集"""
    
    def __init__(self):
        super().__init__("../iris_train.data", 3, 118, 4)
        self.INPUT_TEST_NAME = "../iris_test.data"
        self.INPUT_TEST_NUM = 30
        self.load_train_data()
    
    def load_train_data(self):
        """加载训练数据"""
        self.input_attrs = np.zeros((self.INPUT_LINE_NUM, self.INPUT_ATTRIBUTE_NUM))
        self.input_class = np.zeros(self.INPUT_LINE_NUM, dtype=int)
        
        class_map = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }
        
        with open(self.INPUT_TRAIN_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_LINE_NUM:
                    break
                parts = line.strip().split(',')
                # 格式: sepal_length, sepal_width, petal_length, petal_width, class
                self.input_attrs[i] = [float(x) for x in parts[:4]]
                self.input_class[i] = class_map[parts[4]]
        
        print(f"Loaded {self.INPUT_LINE_NUM} training samples from {self.INPUT_TRAIN_NAME}")
    
    def load_test_data(self):
        """加载测试数据"""
        test_attrs = np.zeros((self.INPUT_TEST_NUM, self.INPUT_ATTRIBUTE_NUM))
        test_class = np.zeros(self.INPUT_TEST_NUM, dtype=int)
        
        class_map = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }
        
        with open(self.INPUT_TEST_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_TEST_NUM:
                    break
                parts = line.strip().split(',')
                test_attrs[i] = [float(x) for x in parts[:4]]
                test_class[i] = class_map[parts[4]]
        
        print(f"Loaded {self.INPUT_TEST_NUM} test samples from {self.INPUT_TEST_NAME}")
        return test_attrs, test_class


class HabermanSurvival(DatasetAbstract):
    """Haberman生存数据集"""
    
    def __init__(self):
        super().__init__("../haberman_train.data", 2, 227, 3)
        self.INPUT_TEST_NAME = "../haberman_test.data"
        self.INPUT_TEST_NUM = 62
        self.load_train_data()
    
    def load_train_data(self):
        """加载训练数据"""
        self.input_attrs = np.zeros((self.INPUT_LINE_NUM, self.INPUT_ATTRIBUTE_NUM))
        self.input_class = np.zeros(self.INPUT_LINE_NUM, dtype=int)
        
        with open(self.INPUT_TRAIN_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_LINE_NUM:
                    break
                parts = line.strip().split(',')
                # 格式: age, year, nodes, survival (1或2)
                self.input_attrs[i] = [float(x) for x in parts[:3]]
                self.input_class[i] = int(parts[3]) - 1  # 转换为0和1
        
        print(f"Loaded {self.INPUT_LINE_NUM} training samples from {self.INPUT_TRAIN_NAME}")
    
    def load_test_data(self):
        """加载测试数据"""
        test_attrs = np.zeros((self.INPUT_TEST_NUM, self.INPUT_ATTRIBUTE_NUM))
        test_class = np.zeros(self.INPUT_TEST_NUM, dtype=int)
        
        with open(self.INPUT_TEST_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_TEST_NUM:
                    break
                parts = line.strip().split(',')
                test_attrs[i] = [float(x) for x in parts[:3]]
                test_class[i] = int(parts[3]) - 1  # 转换为0和1
        
        print(f"Loaded {self.INPUT_TEST_NUM} test samples from {self.INPUT_TEST_NAME}")
        return test_attrs, test_class


class Zoo(DatasetAbstract):
    """Zoo（动物园）数据集"""
    
    def __init__(self):
        super().__init__("../zoo_train.data", 7, 80, 16)
        self.INPUT_TEST_NAME = "../zoo_test.data"
        self.INPUT_TEST_NUM = 21
        self.load_train_data()
    
    def load_train_data(self):
        """加载训练数据"""
        self.input_attrs = np.zeros((self.INPUT_LINE_NUM, self.INPUT_ATTRIBUTE_NUM))
        self.input_class = np.zeros(self.INPUT_LINE_NUM, dtype=int)
        
        with open(self.INPUT_TRAIN_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_LINE_NUM:
                    break
                parts = line.strip().split(',')
                # 跳过动物名称，读取16个属性和1个类别标签
                self.input_attrs[i] = [float(x) for x in parts[1:17]]
                self.input_class[i] = int(parts[17]) - 1  # 转换为0-6
        
        print(f"Loaded {self.INPUT_LINE_NUM} training samples from {self.INPUT_TRAIN_NAME}")
    
    def load_test_data(self):
        """加载测试数据"""
        test_attrs = np.zeros((self.INPUT_TEST_NUM, self.INPUT_ATTRIBUTE_NUM))
        test_class = np.zeros(self.INPUT_TEST_NUM, dtype=int)
        
        with open(self.INPUT_TEST_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_TEST_NUM:
                    break
                parts = line.strip().split(',')
                test_attrs[i] = [float(x) for x in parts[1:17]]
                test_class[i] = int(parts[17]) - 1  # 转换为0-6
        
        print(f"Loaded {self.INPUT_TEST_NUM} test samples from {self.INPUT_TEST_NAME}")
        return test_attrs, test_class
