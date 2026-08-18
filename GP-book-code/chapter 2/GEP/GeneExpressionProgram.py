# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 2 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).


from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from random import random, randint, shuffle
from typing import List, Tuple, Optional, Callable

from Chromosome import Chromosome


class GeneExpressionProgram:
    """
    基因表达式编程算法主类
    实现了GEP的完整进化流程，包括选择、变异、重组等操作
    """

    ### 超参数 ###
    NUM_RUNS = 5  # 运行次数（用于统计）
    NUM_GENERATIONS = 500  # 最大进化代数
    POPULATION_SIZE = 100  # 种群大小
    NUM_FITNESS_CASES = 10  # 适应度案例数量
    ERROR_TOLERANCE = 0.0000001  # 误差容忍度

    ### 繁殖参数 ###
    MUTATION_RATE = 0.051  # 变异率
    ONE_POINT_CROSSOVER_RATE, TWO_POINT_CROSSOVER_RATE, GENE_CROSSOVER_RATE = 0.2, 0.5, 0.1  # 各种重组率
    IS_TRANSPOSITION_RATE, IS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]  # IS转座参数
    RIS_TRANSPOSITION_RATE, RIS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]  # RIS转座参数
    GENE_TRANSPOSITION_RATE = 0.1  # 基因转座率

    ### 适应度评估参数 ###
    OBJECTIVE_FUNCTION: Optional[Callable] = None  # 目标函数
    FITNESS_FUNCTION: Optional[Callable] = None  # 适应度函数
    FITNESS_FUNCTION_ARGS: List = []  # 适应度函数参数
    OBJECTIVE_MIN: Optional[float] = None  # 目标函数最小值
    OBJECTIVE_MAX: Optional[float] = None  # 目标函数最大值
    FUNCTION_Y_RANGE: Optional[float] = None  # 函数Y值范围

    def __init__(self):
        """初始化基因表达式编程算法"""
        pass

    @staticmethod
    def evolve() -> Tuple[Chromosome, List[float], List[float]]:
        """
        执行基因表达式编程算法
        Returns:
            tuple: 包含以下三个元素的元组
                - 最佳适应染色体
                - 每代平均适应度列表（用于绘图）
                - 每代最佳适应度列表（用于绘图）
        """
        # 检查适应度函数是否设置
        if GeneExpressionProgram.FITNESS_FUNCTION is None:
            raise ValueError("FITNESS_FUNCTION must be set before calling evolve()")

        # 创建初始种群
        population = [Chromosome.generate_random_individual()
                      for _ in range(GeneExpressionProgram.POPULATION_SIZE)]

        generation = 0
        best_fit_individual = None
        average_fitness_by_generation = []
        best_fitness_by_generation = []

        # 主进化循环
        while generation < GeneExpressionProgram.NUM_GENERATIONS:

            ### 评估阶段 ###

            # 计算种群中所有个体的适应度
            try:
                population_fitnesses = GeneExpressionProgram.FITNESS_FUNCTION(
                    *GeneExpressionProgram.FITNESS_FUNCTION_ARGS, *population)
            except Exception as e:
                print(f"Error evaluating population fitness: {e}")
                # 如果评估失败，使用随机适应度值继续
                population_fitnesses = np.random.random(GeneExpressionProgram.POPULATION_SIZE)

            # 找到当前代的最佳个体
            try:
                best_fit_generation = population[np.argmax(population_fitnesses)]
                if generation == 0 or best_fit_individual.fitness() < best_fit_generation.fitness():
                    best_fit_individual = deepcopy(best_fit_generation)
            except (AttributeError, IndexError) as e:
                print(f"Error selecting best individual: {e}")
                # 如果选择失败，随机选择一个个体作为最佳
                best_fit_generation = population[0]
                if generation == 0:
                    best_fit_individual = deepcopy(best_fit_generation)

            # 如果找到最优解，提前终止
            try:
                if (best_fit_individual is not None and
                        Chromosome.max_fitness is not None and
                        abs(best_fit_individual.fitness() - Chromosome.max_fitness) <= GeneExpressionProgram.ERROR_TOLERANCE):
                    average_fitness_generation = float(np.mean(population_fitnesses))
                    average_fitness_by_generation.append(average_fitness_generation)
                    best_fitness_by_generation.append(best_fit_individual.fitness())
                    print(f"Optimal solution found at generation {generation}")
                    break
            except Exception as e:
                print(f"Error checking optimal solution: {e}")

            next_generation = []

            ### 选择阶段（轮盘赌选择 + 简单精英保留）###

            # 将最佳个体直接复制到下一代（精英保留）
            if best_fit_individual is not None:
                next_generation.append(deepcopy(best_fit_individual))

            # 使用轮盘赌选择选择其余父代
            try:
                all_parents = list(GeneExpressionProgram.roulette_wheel_selection(population, len(population)))
            except Exception as e:
                print(f"Error in roulette wheel selection: {e}")
                # 如果选择失败，使用随机选择
                all_parents = list(np.random.choice(population, size=len(population), replace=True))

            # 遗传操作
            try:
                # 变异操作
                all_parents = list(map(GeneExpressionProgram.mutate, all_parents))

                # IS转座
                all_parents = list(map(GeneExpressionProgram.is_transposition, all_parents))

                # RIS转座
                all_parents = list(map(GeneExpressionProgram.ris_transposition, all_parents))

                # 基因转座
                all_parents = list(map(GeneExpressionProgram.gene_transposition, all_parents))
            except Exception as e:
                print(f"Error in genetic operations: {e}")

            # 重组操作
            try:
                shuffle(all_parents)  # 打乱父代顺序
                for i in range(1, GeneExpressionProgram.POPULATION_SIZE, 2):

                    # 处理奇数种群情况，避免索引错误
                    if i + 1 >= GeneExpressionProgram.POPULATION_SIZE:
                        next_generation.append(all_parents[i])
                        break

                    child1, child2 = all_parents[i], all_parents[i + 1]

                    # 单点重组
                    if random() < GeneExpressionProgram.ONE_POINT_CROSSOVER_RATE:
                        child1, child2 = GeneExpressionProgram.one_point_recombination(child1, child2)

                    # 两点重组
                    elif random() < GeneExpressionProgram.TWO_POINT_CROSSOVER_RATE:
                        child1, child2 = GeneExpressionProgram.two_point_recombination(child1, child2)

                    # 基因重组
                    elif random() < GeneExpressionProgram.GENE_CROSSOVER_RATE:
                        child1, child2 = GeneExpressionProgram.gene_recombination(child1, child2)

                    # 将子代加入下一代
                    next_generation.append(child1)
                    next_generation.append(child2)
            except Exception as e:
                print(f"Error in recombination: {e}")

            # 准备下一次迭代
            population = next_generation
            generation += 1

            # 记录统计信息
            try:
                average_fitness_generation = float(np.mean(population_fitnesses))
                average_fitness_by_generation.append(average_fitness_generation)
                if best_fit_individual is not None:
                    best_fitness_by_generation.append(best_fit_individual.fitness())
                else:
                    best_fitness_by_generation.append(0.0)

                print(f"Generation: {generation}\tPopulation Size: {len(population)}\t"
                      f"Average Fitness: {average_fitness_generation:.5f}\t"
                      f"Best Fitness (overall): {best_fit_individual.fitness() if best_fit_individual else 0:.5f}")
            except Exception as e:
                print(f"Error recording statistics: {e}")
                average_fitness_by_generation.append(0.0)
                best_fitness_by_generation.append(0.0)

        return best_fit_individual, average_fitness_by_generation, best_fitness_by_generation

    @staticmethod
    def random_search(num_generations: int, fitness_function: Callable, fitness_function_args: List) -> Tuple[
        Chromosome, List[float], List[float]]:
        """
        随机搜索算法，作为对比基准

        Returns:
            tuple: 最佳个体，平均适应度列表，最佳适应度列表
        """
        best = None
        best_fitness = 0
        average_fitnesses = []
        best_fitnesses = []

        for gen in range(num_generations):
            generation_fitnesses = []
            for _ in range(GeneExpressionProgram.POPULATION_SIZE):
                current = Chromosome.generate_random_individual()
                try:
                    current_fitness = fitness_function(*fitness_function_args, current)
                except Exception as e:
                    print(f"Error evaluating fitness in random search: {e}")
                    current_fitness = 0

                generation_fitnesses.append(current_fitness)
                if best is None or best_fitness <= current_fitness:
                    best = deepcopy(current)
                    best_fitness = best.fitness()

            average_fitnesses.append(np.mean(generation_fitnesses))
            best_fitnesses.append(best_fitness)

            # 如果找到最优解，提前终止
            if Chromosome.max_fitness is not None and best_fitness >= Chromosome.max_fitness:
                print(f"Random search found optimal solution at generation {gen}")
                break

        return best, average_fitnesses, best_fitnesses

    @staticmethod
    def roulette_wheel_selection(chromosomes: List[Chromosome], n: int):
        """
        轮盘赌选择
        从染色体列表中根据适应度比例选择n个个体

        """
        if not chromosomes:
            raise ValueError("Chromosome list is empty")

        # 计算适应度总和
        fitness_values = [c.fitness() for c in chromosomes]
        total = float(sum(fitness_values))

        # 如果适应度总和为0，使用均匀分布
        if total == 0:
            for _ in range(n):
                yield chromosomes[randint(0, len(chromosomes) - 1)]
            return

        i = 0
        fitness = chromosomes[0].fitness()
        while n:
            x = total * (1 - random() ** (1.0 / n))
            total -= x
            while x > fitness:
                x -= fitness
                i += 1
                if i >= len(chromosomes):
                    i = 0
                fitness = chromosomes[i].fitness()
            fitness -= x
            yield chromosomes[i]
            n -= 1

    @staticmethod
    def mutate(chromosome: Chromosome) -> Chromosome:
        """
        变异操作：随机改变染色体中的基因

        Returns:
            变异后的染色体
        """
        head_characters = list(Chromosome.functions.keys()) + Chromosome.terminals

        new_genes = []
        # 对每个基因进行变异
        for gene_idx in range(Chromosome.num_genes):
            new_gene = ""
            # 对基因中的每个字符
            for i in range(len(chromosome.genes[gene_idx])):
                # 根据变异率决定是否变异
                if random() < GeneExpressionProgram.MUTATION_RATE:
                    # 头部可以包含函数和终止符
                    if i < Chromosome.head_length:
                        new_gene += head_characters[randint(0, len(head_characters) - 1)]
                    else:
                        # 尾部只能包含终止符
                        new_gene += Chromosome.terminals[randint(0, len(Chromosome.terminals) - 1)]
                else:
                    new_gene += chromosome.genes[gene_idx][i]
            new_genes.append(new_gene)

        # 创建新染色体以确保重新计算缓存的适应度值
        new_chromosome = Chromosome(new_genes)
        new_chromosome.ephemeral_random_constants = deepcopy(chromosome.ephemeral_random_constants)

        # 变异临时随机常数
        for constant_idx in range(len(new_chromosome.ephemeral_random_constants)):
            if random() < GeneExpressionProgram.MUTATION_RATE:
                new_chromosome.ephemeral_random_constants[constant_idx] = np.random.uniform(
                    *Chromosome.ephemeral_random_constants_range)

        return new_chromosome

    @staticmethod
    def is_transposition(chromosome: Chromosome) -> Chromosome:
        """
        IS转座（插入序列转座）
        将一段基因序列插入到染色体的其他位置

        Returns:
            转座后的染色体
        """
        if random() < GeneExpressionProgram.IS_TRANSPOSITION_RATE:
            # 确定转座参数
            length = np.random.choice(GeneExpressionProgram.IS_ELEMENTS_LENGTH)
            source_gene = randint(0, len(chromosome.genes) - 1)  # 源基因
            target_gene = randint(0, len(chromosome.genes) - 1)  # 目标基因

            # 确保目标位置有效
            max_target_position = max(1, Chromosome.head_length - length)
            target_position = randint(1, max_target_position)  # 目标位置

            sequence_start = randint(0, len(chromosome.genes[source_gene]) - 1)  # 序列起始位置

            transposition_string = chromosome.genes[source_gene][
                                   sequence_start:min(Chromosome.length, sequence_start + length)]

            # 执行替换
            new_chromosome = Chromosome(deepcopy(chromosome.genes))
            new_chromosome.ephemeral_random_constants = deepcopy(chromosome.ephemeral_random_constants)

            # 确保替换不会导致字符串越界
            target_gene_str = new_chromosome.genes[target_gene]
            if target_position + length <= len(target_gene_str):
                new_chromosome.genes[target_gene] = (target_gene_str[:target_position] +
                                                     transposition_string +
                                                     target_gene_str[target_position + length:])
            else:
                # 如果越界，只替换到字符串末尾
                new_chromosome.genes[target_gene] = target_gene_str[:target_position] + transposition_string

            return new_chromosome
        else:
            return chromosome

    @staticmethod
    def ris_transposition(chromosome: Chromosome) -> Chromosome:
        """
        RIS转座（根插入序列转座）
        将函数序列插入到基因头部
        Returns:
            转座后的染色体
        """
        start_point = randint(0, Chromosome.head_length - 1)
        gene = randint(0, Chromosome.num_genes - 1)

        # 寻找函数起始位置
        while (start_point < Chromosome.head_length and
               chromosome.genes[gene][start_point] not in Chromosome.functions):
            start_point += 1

        if (random() < GeneExpressionProgram.RIS_TRANSPOSITION_RATE and
                start_point < Chromosome.head_length and
                chromosome.genes[gene][start_point] in Chromosome.functions):

            ris_length = np.random.choice(GeneExpressionProgram.RIS_ELEMENTS_LENGTH)
            # 确保不会越界
            actual_length = min(ris_length, Chromosome.head_length - start_point)
            ris_string = chromosome.genes[gene][start_point:start_point + actual_length]

            new_chromosome = Chromosome(deepcopy(chromosome.genes))
            new_chromosome.ephemeral_random_constants = deepcopy(chromosome.ephemeral_random_constants)
            old_head = new_chromosome.genes[gene][:Chromosome.head_length]
            new_head = old_head[:start_point] + ris_string + old_head[start_point:]
            # 确保头部长度不变
            new_chromosome.genes[gene] = new_head[:Chromosome.head_length] + new_chromosome.genes[gene][
                                                                             Chromosome.head_length:]

            return new_chromosome
        else:
            return chromosome

    @staticmethod
    def gene_transposition(chromosome: Chromosome) -> Chromosome:
        """
        基因转座
        交换染色体中基因的位置（仅适用于多基因染色体）
        Returns:
            转座后的染色体
        """
        if Chromosome.num_genes > 1 and random() < GeneExpressionProgram.GENE_TRANSPOSITION_RATE:
            # 交换第一个基因和随机选择的基因
            index = randint(0, Chromosome.num_genes - 1)
            new_genes = deepcopy(chromosome.genes)
            temp = new_genes[index]
            new_genes[index] = new_genes[0]
            new_genes[0] = temp
            new_chromosome = Chromosome(new_genes)
            new_chromosome.ephemeral_random_constants = deepcopy(chromosome.ephemeral_random_constants)
            return new_chromosome
        else:
            return chromosome

    @staticmethod
    def one_point_recombination(chromosome1: Chromosome, chromosome2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        单点重组
        在随机选择的位置交换两个染色体的基因内容
        Returns:
            子代1, 子代2
        """
        gene = randint(0, Chromosome.num_genes - 1)
        position = randint(0, Chromosome.length)

        child1_split_gene = chromosome1.genes[gene][:position] + chromosome2.genes[gene][position:]
        child2_split_gene = chromosome2.genes[gene][:position] + chromosome1.genes[gene][position:]

        child1_genes = (chromosome1.genes[:gene] + [child1_split_gene] +
                        (chromosome2.genes[gene + 1:] if gene < Chromosome.num_genes - 1 else []))
        child2_genes = (chromosome2.genes[:gene] + [child2_split_gene] +
                        (chromosome1.genes[gene + 1:] if gene < Chromosome.num_genes - 1 else []))

        child1, child2 = Chromosome(child1_genes), Chromosome(child2_genes)

        # 重组临时随机常数
        constants_split_position = randint(0, Chromosome.length - 1)
        child1.ephemeral_random_constants = (chromosome1.ephemeral_random_constants[:constants_split_position] +
                                             chromosome2.ephemeral_random_constants[constants_split_position:])
        child2.ephemeral_random_constants = (chromosome2.ephemeral_random_constants[:constants_split_position] +
                                             chromosome1.ephemeral_random_constants[constants_split_position:])

        return child1, child2

    @staticmethod
    def two_point_recombination(chromosome1: Chromosome, chromosome2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        两点重组
        在两个随机选择的位置之间交换基因内容
        Returns:
            子代1, 子代2
        """
        # 生成两个交叉点
        position1, position2 = sorted([randint(0, Chromosome.length * Chromosome.num_genes - 1),
                                       randint(0, Chromosome.length * Chromosome.num_genes - 1)])

        # 将基因连接成单个字符串以便操作
        child1_genes_str = "".join(chromosome1.genes)
        child2_genes_str = "".join(chromosome2.genes)

        # 执行交叉
        child1_genes = child1_genes_str[:position1] + child2_genes_str[position1:position2] + child1_genes_str[
                                                                                              position2:]
        child2_genes = child2_genes_str[:position1] + child1_genes_str[position1:position2] + child2_genes_str[
                                                                                              position2:]

        # 将字符串分割回基因列表
        child1_genes = [child1_genes[i:i + Chromosome.length]
                        for i in range(0, Chromosome.num_genes * Chromosome.length, Chromosome.length)]
        child2_genes = [child2_genes[i:i + Chromosome.length]
                        for i in range(0, Chromosome.num_genes * Chromosome.length, Chromosome.length)]

        child1, child2 = Chromosome(child1_genes), Chromosome(child2_genes)

        # 重组临时随机常数
        split_positions = sorted([randint(0, Chromosome.length - 1), randint(0, Chromosome.length - 1)])
        child1.ephemeral_random_constants = (chromosome1.ephemeral_random_constants[:split_positions[0]] +
                                             chromosome2.ephemeral_random_constants[
                                             split_positions[0]:split_positions[1]] +
                                             chromosome1.ephemeral_random_constants[split_positions[1]:])
        child2.ephemeral_random_constants = (chromosome2.ephemeral_random_constants[:split_positions[0]] +
                                             chromosome1.ephemeral_random_constants[
                                             split_positions[0]:split_positions[1]] +
                                             chromosome2.ephemeral_random_constants[split_positions[1]:])
        return child1, child2

    @staticmethod
    def gene_recombination(chromosome1: Chromosome, chromosome2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        基因重组
        交换两个染色体中的整个基因（适用于多基因染色体）
        Returns:
            子代1, 子代2
        """
        # 选择要交换的基因
        gene = randint(0, Chromosome.num_genes - 1)

        # 初始化子代基因
        child1_genes = deepcopy(chromosome1.genes)
        child2_genes = deepcopy(chromosome2.genes)

        # 执行交换
        child1_genes[gene] = chromosome2.genes[gene]
        child2_genes[gene] = chromosome1.genes[gene]

        return Chromosome(child1_genes), Chromosome(child2_genes)

    @staticmethod
    def plot_reps(avg_fitnesses: List[List[float]], best_fitnesses: List[List[float]],
                  random_search_avg: Optional[List[float]] = None,
                  random_search_best: Optional[List[float]] = None) -> None:
        """
        绘制多次运行的统计结果

        """
        is_random_search = not (random_search_avg is None or random_search_best is None)

        plt.subplots(1, 2, figsize=(16, 8))

        # 左图：平均适应度
        plt.subplot(1, 2, 1)
        plt.title("Average Fitness by Generation")
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness")

        # 绘制每次运行
        for rep in range(len(avg_fitnesses)):
            plt.plot(range(len(avg_fitnesses[rep])), avg_fitnesses[rep], label=f"Rep {rep + 1} Average")

        if is_random_search:
            plt.plot(range(len(random_search_avg)), random_search_avg, label="Random Search Average")

        plt.legend(loc="upper left")

        # 右图：最佳适应度
        plt.subplot(1, 2, 2)
        plt.title("Best Fitness by Generation")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")

        # 绘制每次运行
        for rep in range(len(best_fitnesses)):
            plt.plot(range(len(best_fitnesses[rep])), best_fitnesses[rep], label=f"Rep {rep + 1} Best")

        if is_random_search:
            plt.plot(range(len(random_search_best)), random_search_best, label="Random Search Best")

        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()