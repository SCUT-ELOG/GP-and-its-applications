# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 7 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
MONK数据集实现
从C++原项目转换而来
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_abstract import DatasetAbstract


class Monk1(DatasetAbstract):
    """MONK-1数据集"""
    
    def __init__(self):
        super().__init__("../monks-1.train", 2, 124, 15)
        self.INPUT_TEST_NAME = "../monks-1.test"
        self.INPUT_TEST_NUM = 432
        self.load_train_data()
    
    def load_train_data(self):
        """加载训练数据"""
        self.input_attrs = np.zeros((self.INPUT_LINE_NUM, self.INPUT_ATTRIBUTE_NUM))
        self.input_class = np.zeros(self.INPUT_LINE_NUM, dtype=int)
        
        with open(self.INPUT_TRAIN_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_LINE_NUM:
                    break
                parts = line.strip().split()
                # 格式: class a1 a2 a3 a4 a5 a6 id
                self.input_class[i] = int(parts[0])
                a1, a2, a3, a4, a5, a6 = map(int, parts[1:7])
                
                # One-hot编码
                self.input_attrs[i, a1 - 1] = 1  # a1: 1-3
                self.input_attrs[i, a2 + 2] = 1  # a2: 1-3, offset by 3
                self.input_attrs[i, 6] = a3 - 1  # a3: 1-2
                self.input_attrs[i, a4 + 6] = 1  # a4: 1-3, offset by 7
                self.input_attrs[i, a5 + 9] = 1  # a5: 1-4, offset by 10
                self.input_attrs[i, 14] = a6 - 1  # a6: 1-2
        
        print(f"Loaded {self.INPUT_LINE_NUM} training samples from {self.INPUT_TRAIN_NAME}")
    
    def load_test_data(self):
        """加载测试数据"""
        test_attrs = np.zeros((self.INPUT_TEST_NUM, self.INPUT_ATTRIBUTE_NUM))
        test_class = np.zeros(self.INPUT_TEST_NUM, dtype=int)
        
        with open(self.INPUT_TEST_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_TEST_NUM:
                    break
                parts = line.strip().split()
                test_class[i] = int(parts[0])
                a1, a2, a3, a4, a5, a6 = map(int, parts[1:7])
                
                # One-hot编码
                test_attrs[i, a1 - 1] = 1
                test_attrs[i, a2 + 2] = 1
                test_attrs[i, 6] = a3 - 1
                test_attrs[i, a4 + 6] = 1
                test_attrs[i, a5 + 9] = 1
                test_attrs[i, 14] = a6 - 1
        
        print(f"Loaded {self.INPUT_TEST_NUM} test samples from {self.INPUT_TEST_NAME}")
        return test_attrs, test_class


class Monk2(DatasetAbstract):
    """MONK-2数据集"""
    
    def __init__(self):
        super().__init__("../monks-2.train", 2, 169, 15)
        self.INPUT_TEST_NAME = "../monks-2.test"
        self.INPUT_TEST_NUM = 432
        self.load_train_data()
    
    def load_train_data(self):
        """加载训练数据"""
        self.input_attrs = np.zeros((self.INPUT_LINE_NUM, self.INPUT_ATTRIBUTE_NUM))
        self.input_class = np.zeros(self.INPUT_LINE_NUM, dtype=int)
        
        with open(self.INPUT_TRAIN_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_LINE_NUM:
                    break
                parts = line.strip().split()
                self.input_class[i] = int(parts[0])
                a1, a2, a3, a4, a5, a6 = map(int, parts[1:7])
                
                # One-hot编码
                self.input_attrs[i, a1 - 1] = 1
                self.input_attrs[i, a2 + 2] = 1
                self.input_attrs[i, 6] = a3 - 1
                self.input_attrs[i, a4 + 6] = 1
                self.input_attrs[i, a5 + 9] = 1
                self.input_attrs[i, 14] = a6 - 1
        
        print(f"Loaded {self.INPUT_LINE_NUM} training samples from {self.INPUT_TRAIN_NAME}")
    
    def load_test_data(self):
        """加载测试数据"""
        test_attrs = np.zeros((self.INPUT_TEST_NUM, self.INPUT_ATTRIBUTE_NUM))
        test_class = np.zeros(self.INPUT_TEST_NUM, dtype=int)
        
        with open(self.INPUT_TEST_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_TEST_NUM:
                    break
                parts = line.strip().split()
                test_class[i] = int(parts[0])
                a1, a2, a3, a4, a5, a6 = map(int, parts[1:7])
                
                # One-hot编码
                test_attrs[i, a1 - 1] = 1
                test_attrs[i, a2 + 2] = 1
                test_attrs[i, 6] = a3 - 1
                test_attrs[i, a4 + 6] = 1
                test_attrs[i, a5 + 9] = 1
                test_attrs[i, 14] = a6 - 1
        
        print(f"Loaded {self.INPUT_TEST_NUM} test samples from {self.INPUT_TEST_NAME}")
        return test_attrs, test_class


class Monk3(DatasetAbstract):
    """MONK-3数据集"""
    
    def __init__(self):
        super().__init__("../monks-3.train", 2, 122, 15)
        self.INPUT_TEST_NAME = "../monks-3.test"
        self.INPUT_TEST_NUM = 432
        self.load_train_data()
    
    def load_train_data(self):
        """加载训练数据"""
        self.input_attrs = np.zeros((self.INPUT_LINE_NUM, self.INPUT_ATTRIBUTE_NUM))
        self.input_class = np.zeros(self.INPUT_LINE_NUM, dtype=int)
        
        with open(self.INPUT_TRAIN_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_LINE_NUM:
                    break
                parts = line.strip().split()
                self.input_class[i] = int(parts[0])
                a1, a2, a3, a4, a5, a6 = map(int, parts[1:7])
                
                # One-hot编码
                self.input_attrs[i, a1 - 1] = 1
                self.input_attrs[i, a2 + 2] = 1
                self.input_attrs[i, 6] = a3 - 1
                self.input_attrs[i, a4 + 6] = 1
                self.input_attrs[i, a5 + 9] = 1
                self.input_attrs[i, 14] = a6 - 1
        
        print(f"Loaded {self.INPUT_LINE_NUM} training samples from {self.INPUT_TRAIN_NAME}")
    
    def load_test_data(self):
        """加载测试数据"""
        test_attrs = np.zeros((self.INPUT_TEST_NUM, self.INPUT_ATTRIBUTE_NUM))
        test_class = np.zeros(self.INPUT_TEST_NUM, dtype=int)
        
        with open(self.INPUT_TEST_NAME, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.INPUT_TEST_NUM:
                    break
                parts = line.strip().split()
                test_class[i] = int(parts[0])
                a1, a2, a3, a4, a5, a6 = map(int, parts[1:7])
                
                # One-hot编码
                test_attrs[i, a1 - 1] = 1
                test_attrs[i, a2 + 2] = 1
                test_attrs[i, 6] = a3 - 1
                test_attrs[i, a4 + 6] = 1
                test_attrs[i, a5 + 9] = 1
                test_attrs[i, 14] = a6 - 1
        
        print(f"Loaded {self.INPUT_TEST_NUM} test samples from {self.INPUT_TEST_NAME}")
        return test_attrs, test_class
