# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 4 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import numpy as np
from typing import List
import random

# 个体类，包含子染色体、各项系数和适应度值
class Individual:
    def __init__(self, chromosomes: List[List[float]], coefficients: List[float]):
        self.chromosomes = chromosomes
        self.coefficients = coefficients
        self.fitness = float('inf')

# 用于随机初始化桶B_w
def initialize_bucket(size: int) -> List[float]:
    return [random.random() for _ in range(size)]

# 用于从桶B_w中随机选择子染色体C_w
def select_chromosome(bucket: List[float], size: int) -> List[float]:
    return random.sample(bucket, size)

# 用于计算系数矩阵X
def evaluate_chromosome(chromosome: List[float], data: np.ndarray) -> np.ndarray:
    # 使染色体数组和输入数据的维度相同，以正确进行逐元素运算
    n_features = data.shape[1] - 1  # 去除目标列
    if len(chromosome) > n_features:
        chromosome = chromosome[:n_features]
    elif len(chromosome) < n_features:
        chromosome = chromosome + [0] * (n_features - len(chromosome))

    return np.sum(data[:, :-1] * np.array(chromosome).reshape(1, -1), axis=1)

# 用于生成特征矩阵X和目标向量y（见公式4-11、4-12）
def generate_matrices(individual: Individual, training_data: np.ndarray) -> tuple:
    # Simplified example - actual implementation would depend on specific problem
    X = np.ones((len(training_data), len(individual.chromosomes) + 1))
    for i, chromosome in enumerate(individual.chromosomes):
        X[:, i + 1] = evaluate_chromosome(chromosome, training_data)
    y = training_data[:, -1].reshape(-1, 1)
    return X, y

# 计算系数β（见公式4-13）
def calculate_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # 使用最小二乘法
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# 归一化系数β（见公式4-14）
def normalize_coefficients(coeffs: np.ndarray) -> np.ndarray:
    # 所有系数的绝对值求和作为分母
    abs_sum = np.sum(np.abs(coeffs))
    normalized_coeffs = coeffs / abs_sum
    return normalized_coeffs

# 简化的CL-FEM评估机制，得到适应度V
def evaluate_fitness(individual: Individual, training_data: np.ndarray) -> float:
    X, y = generate_matrices(individual, training_data)
    y_pred = X @ individual.coefficients
    return np.mean((y - y_pred) ** 2)

# 桶更新函数
def update_buckets(population, buckets, p):
    for w in range(p):
        freq = {}
        # 统计当前种群中第w个染色体的所有符号出现次数
        for individual in population:
            chrom = individual.chromosomes[w]
            for symbol in chrom:
                freq[symbol] = freq.get(symbol, 0) + 1
        # 归一化为概率
        total = sum(freq.values())
        for symbol in freq:
            freq[symbol] /= total
        buckets[w] = freq

# 搜索框架
def search_hidden_function(p: int, chromosome_size: int, bucket_size: int,
                           population_size: int, max_generations: int,
                           training_data: np.ndarray) -> Individual:

    # 初始化桶
    buckets = [initialize_bucket(bucket_size) for _ in range(p)]
    # 初始化种群
    population = []
    for i in range(population_size):
        # 从桶中选择染色体
        chromosomes = [select_chromosome(buckets[w], chromosome_size)
                       for w in range(p)]

        # 计算X,y和归一化后的系数，并得到个体
        X, y = generate_matrices(Individual(chromosomes, []), training_data)
        coeffs = calculate_coefficients(X, y)
        coeffs = normalize_coefficients(coeffs)

        individual = Individual(chromosomes, coeffs)
        individual.fitness = evaluate_fitness(individual, training_data)
        population.append(individual)

    # 进化流程
    generation = 0
    while generation < max_generations:
        for i in range(population_size):
            # 变异和交叉的参数
            # F：缩放因子，CR：交叉率
            F = random.random()
            CR = random.random()

            # 随机选择两个不同个体用于突变
            r1, r2 = random.sample(range(population_size), 2)
            while r1 == i or r2 == i:
                r1, r2 = random.sample(range(population_size), 2)

            # 通过突变和交叉产生试验颜色体
            trial_chromosomes = []
            # 遍历每个染色体
            for w in range(p):
                # 复制第i个个体的第w个染色体
                trial_chrom = population[i].chromosomes[w].copy()
                # 遍历每个基因
                for j in range(chromosome_size):
                    # 随机数小于CR时该位的染色体被替换为对应位突变染色体
                    if random.random() < CR:
                        trial_chrom[j] = (population[i].chromosomes[w][j] +
                                          F * (population[r1].chromosomes[w][j] -
                                               population[r2].chromosomes[w][j]))
                # 修改后的染色体加入试验列表
                trial_chromosomes.append(trial_chrom)

            # 计算试验个体的系数
            X, y = generate_matrices(Individual(trial_chromosomes, []), training_data)
            trial_coeffs = calculate_coefficients(X, y)
            trial_coeffs = normalize_coefficients(trial_coeffs)

            # 评估试验个体
            trial = Individual(trial_chromosomes, trial_coeffs)
            trial.fitness = evaluate_fitness(trial, training_data)

            # 若试验个体评估结果更优，则替换原个体
            if trial.fitness < population[i].fitness:
                population[i] = trial
            # 更新桶
            buckets = [{} for _ in range(p)]
            update_buckets(population, buckets, p)

        generation += 1

    # 返回最优个体
    best_individual = min(population, key=lambda x: x.fitness)
    return best_individual

# 生成在一个曲线上的样本数据作为样例，[-,]之间200个点
def generate_sin_data(n_samples: int = 200) -> np.ndarray:
    x = np.linspace(-np.pi, np.pi, n_samples)
    y = np.sin(x)
    return np.column_stack((x, y))
    np.random.seed(42)
    random.seed(42)

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