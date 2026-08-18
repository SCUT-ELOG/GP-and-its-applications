# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 6 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import random
from typing import List, Union


# 定义树节点类
class Node:
    def __init__(self, value: Union[str, float], children: List['Node'] = None):
        self.value = value
        self.children = children if children is not None else []

    def __repr__(self):
        if self.children:
            return f"({self.value} {' '.join(map(str, self.children))})"
        return str(self.value)


# 基于表6-1的函数集定义
FUNCTION_SET = ['+', '-', '*', 'DIV', 'SQRT', 'LOG', 'MIN', 'MAX']
# 基于表6-2的终端集定义
TERMINAL_SET_TSR = ['CN', 'MRT', 'rank_u', 'rank_oct', 'RN', 'RP']  # TSR终端集
TERMINAL_SET_RSR = ['exec_time', 'EST', 'OCT', 'ROT', 'AT']  # RSR终端集


# 随机生成一个表达式树
def random_expression(max_depth: int = 3, is_tsr: bool = True) -> Node:
    if max_depth <= 0 or (random.random() < 0.5 and max_depth > 1):
        # 生成终端节点
        terminal_set = TERMINAL_SET_TSR if is_tsr else TERMINAL_SET_RSR
        value = random.choice(terminal_set)
        return Node(value)
    else:
        # 生成函数节点
        func = random.choice(FUNCTION_SET)
        # 根据函数类型确定子节点数量
        if func in ['SQRT', 'LOG']:
            num_children = 1
        else:
            num_children = 2  # 其他函数都是二元操作符

        children = [random_expression(max_depth - 1, is_tsr) for _ in range(num_children)]

        return Node(func, children)