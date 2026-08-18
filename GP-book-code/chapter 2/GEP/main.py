# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 2 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

from copy import deepcopy

from Chromosome import Chromosome
from GeneExpressionProgram import GeneExpressionProgram

import numpy as np


def a4_a3_a2_a1():
    # define objective function
    GeneExpressionProgram.OBJECTIVE_FUNCTION = staticmethod(lambda a: a ** 4 + a ** 3 + a ** 2 + a)
    GeneExpressionProgram.OBJECTIVE_MIN, GeneExpressionProgram.OBJECTIVE_MAX = 0, 20

    # Define terminals and functions
    Chromosome.terminals = ["a"]
    Chromosome.functions = {
        "+": {"args": 2, "f": lambda x, y: x + y},
        "-": {"args": 2, "f": lambda x, y: x - y},
        "*": {"args": 2, "f": lambda x, y: x * y},
        "/": {"args": 2, "f": lambda x, y: x / y if y != 0 else 1}  # 添加除零保护
    }

    # 生成适应度案例
    Chromosome.fitness_cases = [
        ({"a": a}, GeneExpressionProgram.OBJECTIVE_FUNCTION(a))
        for a in np.random.rand(GeneExpressionProgram.NUM_FITNESS_CASES) * (
                GeneExpressionProgram.OBJECTIVE_MAX - GeneExpressionProgram.OBJECTIVE_MIN) + GeneExpressionProgram.OBJECTIVE_MIN
    ]

    GeneExpressionProgram.FUNCTION_Y_RANGE = \
        GeneExpressionProgram.OBJECTIVE_FUNCTION(GeneExpressionProgram.OBJECTIVE_MAX) - \
        GeneExpressionProgram.OBJECTIVE_FUNCTION(GeneExpressionProgram.OBJECTIVE_MIN)

    Chromosome.linking_function = "+"

    # 根据选择的适应度函数设置最大适应度
    # 对于 inv_squared_error，最大适应度是 1
    Chromosome.max_fitness = 1

    # 设置适应度函数
    GeneExpressionProgram.FITNESS_FUNCTION = Chromosome.inv_squared_error
    GeneExpressionProgram.FITNESS_FUNCTION_ARGS = []

    # 设置染色体结构参数
    Chromosome.head_length = 8
    Chromosome.length = Chromosome.head_length * 2 + 1  # 根据GEP规则：tail = head*(n-1)+1
    Chromosome.num_genes = 1  # 单基因染色体

    # 修改这里：正确解包返回值
    best_individual, avg_fitnesses, best_fitnesses = GeneExpressionProgram.evolve()
    best_individual.print_tree()

    best_individual.plot_solution(
        objective_function=GeneExpressionProgram.OBJECTIVE_FUNCTION,
        x_min=GeneExpressionProgram.OBJECTIVE_MIN,
        x_max=GeneExpressionProgram.OBJECTIVE_MAX,
        avg_fitnesses=avg_fitnesses,  # 使用返回的平均适应度列表
        best_fitnesses=best_fitnesses,  # 使用返回的最佳适应度列表
        variable_name="a"
    )

    return best_individual, avg_fitnesses, best_fitnesses

if __name__ == "__main__":

    a4_a3_a2_a1();
