# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 4 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import random
import numpy as np
from typing import List, Dict, Tuple


# 个体类定义
class Individual:
    def __init__(self, chromosomes: List[List[float]], coefficients: np.ndarray = None):
        """
        表示一个候选个体（Individual）

        参数：
        - chromosomes: List[List[float]] —— 个体的多个子染色体（C_w）
        - coefficients: np.ndarray —— 系数向量 β
        """
        self.chromosomes = chromosomes
        self.coefficients = coefficients
        self.fitness: float = 0.0


# 阶段 1：初始化种群与桶
def initialize_population(
    p: int,                   # 子染色体数
    chromosome_size: int,     # 每个子染色体长度
    bucket_size: int,         # 每个桶初始容量
    population_size: int,     # 种群大小
    training_data: np.ndarray # 训练数据
) -> Tuple[List[Individual], List[List[float]]]:
    """
    初始化桶 B_w 与种群 Population（对应伪代码初始化部分）
    """
    # 初始化桶：每个桶存储随机符号（浮点数）
    buckets: List[List[float]] = [
        [random.random() for _ in range(bucket_size)] for _ in range(p)
    ]

    population: List[Individual] = []

    # 构造初始种群
    for i in range(population_size):
        chromosomes = [random.sample(buckets[w], chromosome_size) for w in range(p)]

        # 生成 X 和 y
        X = np.ones((len(training_data), p + 1))
        for w in range(p):
            chrom = chromosomes[w]
            chrom = (chrom + [0] * (training_data.shape[1] - 1 - len(chrom)))[: training_data.shape[1] - 1]
            X[:, w + 1] = np.sum(training_data[:, :-1] * np.array(chrom).reshape(1, -1), axis=1)
        y = training_data[:, -1].reshape(-1, 1)

        # 计算并归一化系数 β
        coeffs = np.linalg.pinv(X.T @ X) @ X.T @ y
        coeffs = coeffs / np.sum(np.abs(coeffs))

        # CL-FEM 评估适应度
        individual = Individual(chromosomes, coeffs)
        y_pred = X @ coeffs
        individual.fitness = np.mean((y - y_pred) ** 2)

        population.append(individual)

    return population, buckets


# 阶段 2：进化过程（变异、交叉、选择、桶更新）
def evolve_population(
    population: List[Individual],
    buckets: List[List[float]],
    p: int,
    chromosome_size: int,
    max_generations: int,
    training_data: np.ndarray
) -> List[Individual]:
    """
    进化阶段（对应伪代码 while 循环部分）
    """
    generation = 0
    population_size = len(population)

    while generation < max_generations:

        # 遍历种群个体
        for i in range(population_size):
            F, CR = random.random(), random.random()  # 差分进化参数

            # 随机选择两个不同个体用于变异
            r1, r2 = random.sample(range(population_size), 2)
            while r1 == i or r2 == i:
                r1, r2 = random.sample(range(population_size), 2)

            # 生成试验个体 U_i
            trial_chromosomes: List[List[float]] = []
            for w in range(p):
                base = population[i].chromosomes[w][:]
                for j in range(chromosome_size):
                    if random.random() < CR:
                        base[j] = (
                            population[i].chromosomes[w][j]
                            + F * (population[r1].chromosomes[w][j]
                                   - population[r2].chromosomes[w][j])
                        )
                trial_chromosomes.append(base)

            # 获取试验个体的系数与适应度
            X = np.ones((len(training_data), p + 1))
            for w in range(p):
                chrom = trial_chromosomes[w]
                chrom = (chrom + [0] * (training_data.shape[1] - 1 - len(chrom)))[: training_data.shape[1] - 1]
                X[:, w + 1] = np.sum(training_data[:, :-1] * np.array(chrom).reshape(1, -1), axis=1)
            y = training_data[:, -1].reshape(-1, 1)

            coeffs = np.linalg.pinv(X.T @ X) @ X.T @ y
            coeffs = coeffs / np.sum(np.abs(coeffs))

            trial = Individual(trial_chromosomes, coeffs)
            y_pred = X @ coeffs
            trial.fitness = np.mean((y - y_pred) ** 2)

            # 选择：保留更优个体
            if trial.fitness < population[i].fitness:
                population[i] = trial

        # 更新桶（统计符号出现概率）
        for w in range(p):
            freq: Dict[float, float] = {}
            for ind in population:
                for symbol in ind.chromosomes[w]:
                    freq[symbol] = freq.get(symbol, 0) + 1
            total = sum(freq.values())
            for symbol in freq:
                freq[symbol] /= total
            buckets[w] = freq

        generation += 1

    return population



# 阶段 3：主控函数（整合流程）
def search_hidden_function(
    p: int,
    chromosome_size: int,
    bucket_size: int,
    population_size: int,
    max_generations: int,
    training_data: np.ndarray
) -> Individual:
    """
    主搜索流程控制（对应伪代码主框架）
    -------------------------------------------------
    1. 初始化（initialize_population）
    2. 进化迭代（evolve_population）
    3. 输出最优个体
    -------------------------------------------------
    """
    # 阶段 1：初始化
    population, buckets = initialize_population(
        p, chromosome_size, bucket_size, population_size, training_data
    )

    # 阶段 2：进化迭代
    population = evolve_population(
        population, buckets, p, chromosome_size, max_generations, training_data
    )

    # 阶段 3：输出最优个体
    best_individual: Individual = min(population, key=lambda ind: ind.fitness)
    return best_individual

def generate_sin_data(n_samples: int = 200) -> np.ndarray:
    x = np.linspace(-np.pi, np.pi, n_samples)
    y = np.sin(x)
    return np.column_stack((x, y))


training_data = generate_sin_data(200)

# ---------- 参数设置 ----------
P = 3                 # 子染色体数量
CHROMOSOME_SIZE = 10  # 每个子染色体长度
BUCKET_SIZE = 50
POPULATION_SIZE = 100
MAX_GENERATIONS = 1e3

# ---------- 执行搜索 ----------
best = search_hidden_function(P, CHROMOSOME_SIZE, BUCKET_SIZE,
                                POPULATION_SIZE, MAX_GENERATIONS, training_data)

# ---------- 输出结果 ----------
print("\n=== 搜索结果 ===")
print("最佳适应度:", best.fitness)
print("系数 β：\n", best.coefficients)