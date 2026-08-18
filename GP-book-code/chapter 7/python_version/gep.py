# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 7 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
GEP (Gene Expression Programming) 核心算法实现
从C++原项目转换而来
"""

import numpy as np
import random
from typing import List, Tuple
from math import log2, exp, sqrt, log, sin, cos, fabs

# 算法参数
POPULATION_SIZE = 1000
GENERATION_LIMIT = 2000
CHROMOSOME_SIZE = 100

CROSSOVER_RATE = 0.50
MUTATION_RATE = 0.10
ROTATION_RATE = 0.02

FUNCTION_SET_SIZE = 12  # IF + - * OR AND / SQRT EXP LOG SIN COS
RANDOM_SET_SIZE = 5
RANDOM_SET = [1, 2, 3, 5, 7]


class Chromosome:
    """染色体类"""
    TERMINAL_SET_SIZE = 0  # 将在训练时设置
    
    def __init__(self):
        self.gene = np.zeros(CHROMOSOME_SIZE, dtype=int)
        self.child_position = np.zeros(CHROMOSOME_SIZE, dtype=int)
        self.encoding_genes = 0
        self.fitness = 0.0
        self.is_modified = True
    
    def random_init(self):
        """随机初始化染色体"""
        for i in range(CHROMOSOME_SIZE):
            self.gene[i] = random.randint(0, FUNCTION_SET_SIZE + self.TERMINAL_SET_SIZE - 1)
        self.is_modified = True
    
    def mutation_and_update_is_modified(self):
        """执行变异操作"""
        for i in range(CHROMOSOME_SIZE):
            if random.random() <= MUTATION_RATE:
                self.gene[i] = random.randint(0, FUNCTION_SET_SIZE + self.TERMINAL_SET_SIZE - 1)
                self.is_modified = True
    
    def rotation(self):
        """旋转操作"""
        rotation_point = random.randint(0, CHROMOSOME_SIZE - 3)
        # [0, rotation_point] 移到后面
        rotation_gene = self.gene[:rotation_point + 1].copy()
        # 将 [rotation_point + 1, CHROMOSOME_SIZE - 1] 移到前面
        self.gene[:CHROMOSOME_SIZE - rotation_point - 1] = self.gene[rotation_point + 1:].copy()
        # 将保存的 [0, rotation_point] 移到后面
        self.gene[CHROMOSOME_SIZE - rotation_point - 1:] = rotation_gene
    
    def is_valid(self) -> bool:
        """判断染色体是否有效"""
        self.encoding_genes = 1
        i = 0
        while i != self.encoding_genes and i < CHROMOSOME_SIZE:
            if self.gene[i] >= FUNCTION_SET_SIZE + self.TERMINAL_SET_SIZE:
                print(f"Error: gene[{i}] out of upper bound: {self.gene[i]}")
                return False
            
            if self.gene[i] < FUNCTION_SET_SIZE:
                # 记录子节点位置
                self.child_position[i] = self.encoding_genes
                if self.gene[i] == 0:  # IF (三目运算符)
                    self.encoding_genes += 3
                elif self.gene[i] >= 7:  # SQRT EXP LOG SIN COS (单目运算符)
                    self.encoding_genes += 1
                else:  # + - * OR AND / (双目运算符)
                    self.encoding_genes += 2
            else:
                self.child_position[i] = 0x7fff0000  # 调试标记
            
            i += 1
        
        return i == self.encoding_genes
    
    def decode_gene(self, input_attrs: np.ndarray, input_line_num: int) -> np.ndarray:
        """
        解码染色体为表达式树并计算结果
        
        Args:
            input_attrs: 输入属性矩阵 [样本数, 特征数]
            input_line_num: 样本数
            
        Returns:
            分类结果数组
        """
        # 构建前序遍历序列
        preorder_traversal = []
        current_node = 0
        right_node_stack = []
        
        while True:
            preorder_traversal.append(self.gene[current_node])
            
            if self.gene[current_node] >= FUNCTION_SET_SIZE:
                # 终结符，无子节点
                if not right_node_stack:
                    break
                current_node = right_node_stack.pop()
            else:
                # 函数节点
                if self.gene[current_node] < 7:  # 非单目运算符
                    if self.gene[current_node] == 0:  # IF，有2个右子节点
                        right_node_stack.append(self.child_position[current_node] + 2)
                    right_node_stack.append(self.child_position[current_node] + 1)
                
                current_node = self.child_position[current_node]
        
        # 从右往左计算表达式值
        terminal_stack = []
        
        for i in range(len(preorder_traversal) - 1, -1, -1):
            gene_val = preorder_traversal[i]
            
            if gene_val >= FUNCTION_SET_SIZE:
                # 终结符
                result = np.zeros(input_line_num)
                
                if gene_val >= FUNCTION_SET_SIZE + RANDOM_SET_SIZE:
                    # 输入属性
                    attr_idx = gene_val - FUNCTION_SET_SIZE - RANDOM_SET_SIZE
                    result = input_attrs[:, attr_idx].copy()
                else:
                    # 随机常数
                    constant = RANDOM_SET[gene_val - FUNCTION_SET_SIZE]
                    result[:] = constant
                
                terminal_stack.append(result)
            else:
                # 函数运算
                operand0 = terminal_stack.pop()
                
                if gene_val == 0:  # IF
                    operand1 = terminal_stack.pop()
                    operand2 = terminal_stack[-1]  # 获取但不弹出
                    result = np.where(operand0 > 1e-6, operand1, operand2)
                    terminal_stack[-1] = result  # 替换栈顶
                    
                elif gene_val == 1:  # +
                    operand1 = terminal_stack[-1]  # 获取但不弹出
                    terminal_stack[-1] = operand0 + operand1  # 替换栈顶
                    
                elif gene_val == 2:  # -
                    operand1 = terminal_stack[-1]  # 获取但不弹出
                    terminal_stack[-1] = operand0 - operand1  # 替换栈顶
                    
                elif gene_val == 3:  # *
                    operand1 = terminal_stack[-1]  # 获取但不弹出
                    terminal_stack[-1] = operand0 * operand1  # 替换栈顶
                    
                elif gene_val == 4:  # OR
                    operand1 = terminal_stack[-1]  # 获取但不弹出
                    result = np.where((operand0 > 0) | (operand1 > 0), 1, 0)
                    terminal_stack[-1] = result  # 替换栈顶
                    
                elif gene_val == 5:  # AND
                    operand1 = terminal_stack[-1]  # 获取但不弹出
                    result = np.where((operand0 > 0) & (operand1 > 0), 1, 0)
                    terminal_stack[-1] = result  # 替换栈顶
                    
                elif gene_val == 6:  # /
                    operand1 = terminal_stack[-1]  # 获取但不弹出
                    result = np.where(np.abs(operand1) < 1e-6, 1, operand0 / operand1)
                    terminal_stack[-1] = result  # 替换栈顶
                    
                elif gene_val == 7:  # SQRT
                    terminal_stack.append(np.sqrt(np.abs(operand0)))
                    
                elif gene_val == 8:  # EXP
                    # 匹配C++版本：只限制上界到20
                    operand0 = np.where(operand0 >= 20, 20, operand0)
                    terminal_stack.append(np.exp(operand0))
                    
                elif gene_val == 9:  # LOG
                    result = np.where(np.abs(operand0) < 1e-6, 0, np.log(np.abs(operand0)))
                    terminal_stack.append(result)
                    
                elif gene_val == 10:  # SIN
                    terminal_stack.append(np.sin(operand0))
                    
                elif gene_val == 11:  # COS
                    terminal_stack.append(np.cos(operand0))
        
        return terminal_stack[0]
    
    def copy(self):
        """复制染色体"""
        new_chrom = Chromosome()
        new_chrom.gene = self.gene.copy()
        new_chrom.child_position = self.child_position.copy()
        new_chrom.encoding_genes = self.encoding_genes
        new_chrom.fitness = self.fitness
        new_chrom.is_modified = self.is_modified
        return new_chrom


class Rule:
    """规则类"""
    def __init__(self, chromosome: Chromosome, classification_result: np.ndarray, target_class: int):
        self.chromosome = chromosome
        self.classification_result = classification_result
        self.target_class = target_class


class GEPClassifier:
    """GEP分类器"""
    
    def __init__(self):
        self.ruleset: List[Rule] = []
        self.order: List[int] = []
        self.default_class = 0
    
    def log2_combination(self, n: int, m: int) -> float:
        """计算log2(C(n, m))"""
        ans = 0.0
        for i in range(m):
            ans += log2(n - i) - log2(m - i)
        return ans
    
    def train_one_class(self, class_num: int, input_attrs: np.ndarray, 
                       input_class: np.ndarray, input_line_num: int,
                       input_attribute_num: int, positive_examples_num: List[int]):
        """
        为一个类别训练GEP规则
        
        Args:
            class_num: 目标类别编号
            input_attrs: 训练数据属性
            input_class: 训练数据标签
            input_line_num: 样本数
            input_attribute_num: 特征数
            positive_examples_num: 各类别正例数
        """
        Chromosome.TERMINAL_SET_SIZE = RANDOM_SET_SIZE + input_attribute_num
        
        population = [Chromosome() for _ in range(POPULATION_SIZE)]
        is_removed = np.zeros(input_line_num, dtype=bool)
        is_false_positive = np.zeros(input_line_num, dtype=bool)
        
        P = positive_examples_num[class_num]  # 正例总数
        N = input_line_num - P  # 负例总数
        last_l_theory = 0.0
        l_min = 999999.0
        
        while True:
            max_fitness = -1.0
            max_fitness_index = 0
            max_p, max_n = 0, 0
            fittest_encoding_genes = 0
            current_fittest_classification = None
            
            # 进化循环
            for generation in range(GENERATION_LIMIT):
                for j in range(POPULATION_SIZE):
                    if generation == 0:
                        # 初始化
                        while True:
                            population[j].random_init()
                            if population[j].is_valid():
                                break
                        population[j].is_modified = True
                    
                    if population[j].is_modified:
                        classification_result = population[j].decode_gene(input_attrs, input_line_num)
                        
                        # 计算覆盖的正负例
                        p, n = 0, 0
                        for k in range(input_line_num):
                            if not is_removed[k] and classification_result[k] > 0:
                                if input_class[k] == class_num:
                                    p += 1
                                else:
                                    n += 1
                        
                        # 计算适应度
                        if p + n == 0:
                            population[j].fitness = 0.0
                        else:
                            consig = ((p / (p + n)) - (P / (P + N))) * ((P + N) / N)
                            population[j].fitness = 0.0 if consig < 0 else consig * exp(p / P - 1)
                        
                        if (population[j].fitness > max_fitness or 
                            (population[j].fitness == max_fitness and 
                             population[j].encoding_genes < fittest_encoding_genes)):
                            max_fitness = population[j].fitness
                            max_fitness_index = j
                            max_p = p
                            max_n = n
                            fittest_encoding_genes = population[j].encoding_genes
                            current_fittest_classification = classification_result.copy()
                        
                        population[j].is_modified = False
                
                if generation % 100 == 0:
                    print(f"Generation {generation}, max fitness = {max_fitness:.6f}, "
                          f"max_p = {max_p}, max_n = {max_n}, P = {P}, "
                          f"encoding_genes = {fittest_encoding_genes}")
                
                if abs(max_fitness - 1.0) < 1e-8:
                    break
                
                # 生成新一代
                new_population = []
                for j in range(POPULATION_SIZE):
                    if j == max_fitness_index:
                        new_population.append(population[j])
                        continue
                    
                    # 锦标赛选择（匹配C++版本的不对称选择）
                    mother_idx = self._tournament_selection_mother(population)
                    parent_compare_idx = random.randint(0, len(population) - 1)
                    father_idx = self._tournament_selection_father(population, parent_compare_idx)
                    
                    if random.random() <= CROSSOVER_RATE:
                        # 交叉
                        elitism = 0.05
                        if random.random() < elitism:
                            mother_idx = max_fitness_index
                        elif random.random() < elitism:
                            father_idx = max_fitness_index
                        
                        new_chrom = self._crossover(population[mother_idx], population[father_idx])
                    else:
                        # 选择适应度高的
                        new_chrom = (population[mother_idx] if population[mother_idx].fitness >= 
                                   population[father_idx].fitness else population[father_idx]).copy()
                        new_chrom.is_modified = False
                    
                    # 旋转
                    if random.random() <= ROTATION_RATE:
                        rotation_cycle = 0
                        while rotation_cycle < 100:
                            new_chrom.rotation()
                            if new_chrom.is_valid():
                                break
                            rotation_cycle += 1
                        new_chrom.is_modified = True
                    
                    # 变异
                    mutate = new_chrom.copy()
                    mutate.mutation_and_update_is_modified()
                    while not mutate.is_valid():
                        mutate = new_chrom.copy()
                        mutate.mutation_and_update_is_modified()
                    new_chrom = mutate
                    
                    # 计算新个体适应度
                    if new_chrom.is_modified:
                        classification_result = new_chrom.decode_gene(input_attrs, input_line_num)
                        p, n = 0, 0
                        for k in range(input_line_num):
                            if not is_removed[k] and classification_result[k] > 1e-8:
                                if input_class[k] == class_num:
                                    p += 1
                                else:
                                    n += 1
                        
                        if p + n == 0:
                            new_chrom.fitness = 0.0
                        else:
                            consig = ((p / (p + n)) - (P / (P + N))) * ((P + N) / N)
                            new_chrom.fitness = 0.0 if consig < 0 else consig * exp(p / P - 1)
                    
                    if new_chrom.fitness >= population[j].fitness:
                        new_population.append(new_chrom)
                    else:
                        new_population.append(population[j])
                
                population = new_population
            
            print(f"Stop at generation {generation}, fitness = {max_fitness:.6f}, "
                  f"encoding_genes = {fittest_encoding_genes}")
            
            # 更新is_removed和is_false_positive
            for k in range(input_line_num):
                if is_removed[k]:
                    continue
                if current_fittest_classification[k] > 0:
                    if input_class[k] == class_num:
                        is_removed[k] = True
                    else:
                        is_false_positive[k] = True
            
            # 更新P
            P -= max_p
            false_positive_num = np.sum(is_false_positive)
            
            # 计算MDL
            total_cover_num = (positive_examples_num[class_num] - P) + false_positive_num
            l_exception = (self.log2_combination(total_cover_num, false_positive_num) +
                          self.log2_combination(input_line_num - total_cover_num, P))
            l_theory = last_l_theory + log2(FUNCTION_SET_SIZE + Chromosome.TERMINAL_SET_SIZE) * fittest_encoding_genes
            l_h = 0.5 * l_theory + l_exception
            
            print(f"l_h = {l_h:.2f}, l_min = {l_min:.2f}, total_cover = {total_cover_num}, "
                  f"false_positive = {false_positive_num}")
            
            if l_h >= l_min:
                print("New rule rejected.")
                break
            
            # 添加规则
            l_min = l_h
            last_l_theory = l_theory
            self.ruleset.append(Rule(population[max_fitness_index].copy(), 
                                    current_fittest_classification, class_num))
            print(f"New rule added, rule num: {len(self.ruleset)}")
            
            if P == 0:
                break
    
    def _tournament_selection_mother(self, population: List[Chromosome]) -> int:
        """锦标赛选择母体（使用3个比较对象）"""
        parent_compare = random.randint(0, len(population) - 1)
        mother = random.randint(0, len(population) - 1)
        if population[mother].fitness < population[parent_compare].fitness:
            mother = parent_compare
        return mother
    
    def _tournament_selection_father(self, population: List[Chromosome], parent_compare_idx: int) -> int:
        """锦标赛选择父体（使用2个比较对象）"""
        father = random.randint(0, len(population) - 1)
        if population[father].fitness < population[parent_compare_idx].fitness:
            father = parent_compare_idx
        return father
    
    def _crossover(self, mother: Chromosome, father: Chromosome) -> Chromosome:
        """交叉操作"""
        new_chrom = Chromosome()
        crossover_cycle = 0
        
        if random.random() < 0.5:
            # 单点交叉
            while crossover_cycle < 100:
                cross_point = random.randint(0, CHROMOSOME_SIZE - 2)
                new_chrom.gene[:cross_point + 1] = mother.gene[:cross_point + 1]
                new_chrom.gene[cross_point + 1:] = father.gene[cross_point + 1:]
                if new_chrom.is_valid():
                    break
                crossover_cycle += 1
        else:
            # 双点交叉
            while crossover_cycle < 100:
                cp1 = random.randint(0, CHROMOSOME_SIZE - 2)
                cp2 = random.randint(0, CHROMOSOME_SIZE - 2)
                while cp1 == cp2:
                    cp2 = random.randint(0, CHROMOSOME_SIZE - 2)
                if cp1 > cp2:
                    cp1, cp2 = cp2, cp1
                
                new_chrom.gene[:cp1 + 1] = mother.gene[:cp1 + 1]
                new_chrom.gene[cp1 + 1:cp2 + 1] = father.gene[cp1 + 1:cp2 + 1]
                new_chrom.gene[cp2 + 1:] = mother.gene[cp2 + 1:]
                if new_chrom.is_valid():
                    break
                crossover_cycle += 1
        
        new_chrom.is_modified = True
        return new_chrom
    
    def postpruning(self, input_class: np.ndarray, input_line_num: int,
                   input_class_num: int, positive_examples_num: List[int]):
        """后剪枝"""
        is_removed = np.zeros(input_line_num, dtype=bool)
        self.order = [0] * len(self.ruleset)
        current_order_index = 1
        
        while current_order_index <= len(self.ruleset):
            # 选择适应度最高的规则
            max_fitness = -1.0
            fittest_idx = 0
            
            for idx, rule in enumerate(self.ruleset):
                if self.order[idx] == 0 and rule.chromosome.fitness > max_fitness:
                    max_fitness = rule.chromosome.fitness
                    fittest_idx = idx
            
            if self.ruleset[fittest_idx].chromosome.fitness <= 1e-8:
                print(f"Post pruning: fitness = {self.ruleset[fittest_idx].chromosome.fitness}")
                break
            
            self.order[fittest_idx] = current_order_index
            current_order_index += 1
            
            # 标记已覆盖样本
            for i in range(input_line_num):
                if self.ruleset[fittest_idx].classification_result[i] > 0:
                    is_removed[i] = True
            
            # 重新计算剩余规则的适应度
            for idx, rule in enumerate(self.ruleset):
                if self.order[idx] == 0:
                    P = positive_examples_num[rule.target_class]
                    N = input_line_num - P
                    p, n = 0, 0
                    
                    for k in range(input_line_num):
                        if not is_removed[k] and rule.classification_result[k] > 0:
                            if input_class[k] == rule.target_class:
                                p += 1
                            else:
                                n += 1
                    
                    if p + n == 0:
                        rule.chromosome.fitness = 0.0
                    else:
                        consig = ((p / (p + n)) - (P / (P + N))) * ((P + N) / N)
                        rule.chromosome.fitness = 0.0 if consig < 0 else consig * exp(p / P - 1)
        
        print(f"Rules order: {self.order}")
        
        # 确定默认类别
        unclassified_example_num = [0] * input_class_num
        for k in range(input_line_num):
            if not is_removed[k]:
                unclassified_example_num[input_class[k]] += 1
        
        self.default_class = np.argmax(unclassified_example_num)
        print(f"Default class is {self.default_class}")
    
    def predict(self, test_attrs: np.ndarray, input_test_num: int) -> np.ndarray:
        """预测"""
        test_result_class = np.full(input_test_num, -1, dtype=int)
        
        # 按order顺序应用规则
        valid_rules = [(self.order[i] - 1, rule) for i, rule in enumerate(self.ruleset) 
                      if self.order[i] > 0]
        valid_rules.sort(key=lambda x: x[0])
        
        for _, rule in valid_rules:
            classification = rule.chromosome.decode_gene(test_attrs, input_test_num)
            for j in range(input_test_num):
                if test_result_class[j] == -1 and classification[j] > 0:
                    test_result_class[j] = rule.target_class
        
        # 使用默认类别
        test_result_class[test_result_class == -1] = self.default_class
        
        return test_result_class
