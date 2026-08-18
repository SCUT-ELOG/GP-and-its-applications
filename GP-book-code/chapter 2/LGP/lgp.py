# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 2 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
线性遗传编程(LGP)算法实现 - 改进版本
用于符号回归问题
"""

import random
import math
import numpy as np
from typing import List, Tuple, Callable, Any
import time

# 定义操作符常量
ADD = 0
SUB = 1
MUL = 2
DIV = 3
SIN = 4
COS = 5

# 操作符名称映射
OP_NAMES = {
    ADD: '+',
    SUB: '-',
    MUL: '*',
    DIV: '/',
    SIN: 'sin',
    COS: 'cos'
}

class LGPProgram:
    """LGP程序类，表示一个个体"""

    def __init__(self, instructions: List[Tuple]):
        """
        初始化LGP程序

        Args:
            instructions: 指令列表，每条指令为 (func_id, out_reg, in_reg1, in_reg2)
        """
        self.instructions = instructions
        self.fitness = float('inf')  # 适应度，初始设为无穷大
        self.length = len(instructions)

    def execute(self, inputs: List[float], const_registers: List[float],
                num_calc_registers: int = 5) -> float:
        """
        执行程序并返回输出

        Args:
            inputs: 输入寄存器值
            const_registers: 常数寄存器值
            num_calc_registers: 计算寄存器数量

        Returns:
            输出值（假设为r0寄存器的值）
        """
        # 初始化寄存器：输入寄存器 + 常数寄存器 + 计算寄存器
        registers = inputs.copy() + const_registers.copy() + [0.0] * num_calc_registers

        for instruction in self.instructions:
            func_id, out_reg, in_reg1, in_reg2 = instruction

            # 确保寄存器索引在有效范围内
            if (out_reg >= len(registers) or in_reg1 >= len(registers) or
                in_reg2 >= len(registers)):
                continue

            try:
                if func_id == ADD:
                    registers[out_reg] = registers[in_reg1] + registers[in_reg2]
                elif func_id == SUB:
                    registers[out_reg] = registers[in_reg1] - registers[in_reg2]
                elif func_id == MUL:
                    registers[out_reg] = registers[in_reg1] * registers[in_reg2]
                elif func_id == DIV:
                    # 除法保护，避免除零错误
                    if abs(registers[in_reg2]) < 1e-10:
                        registers[out_reg] = 1.0
                    else:
                        registers[out_reg] = registers[in_reg1] / registers[in_reg2]
                elif func_id == SIN:
                    registers[out_reg] = math.sin(registers[in_reg1])
                elif func_id == COS:
                    registers[out_reg] = math.cos(registers[in_reg1])
            except (ValueError, ZeroDivisionError, OverflowError):
                # 处理数值计算错误
                registers[out_reg] = 0.0

        return registers[0]  # 假设输出为r0

    def __str__(self):
        """返回程序的字符串表示"""
        program_str = []
        for i, (func_id, out_reg, in_reg1, in_reg2) in enumerate(self.instructions):
            if func_id in [SIN, COS]:  # 一元操作符
                program_str.append(f"r{out_reg} = {OP_NAMES[func_id]}(r{in_reg1})")
            else:  # 二元操作符
                program_str.append(f"r{out_reg} = r{in_reg1} {OP_NAMES[func_id]} r{in_reg2}")
        return "\n".join(program_str)

    def copy(self):
        """创建程序的深拷贝"""
        return LGPProgram(self.instructions.copy())

class LGP:
    """线性遗传编程算法主类"""

    def __init__(self,
                 pop_size: int = 100,
                 min_length: int = 5,
                 max_length: int = 20,
                 generations: int = 200,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.05,
                 crossover_rate: float = 0.8,
                 num_input_registers: int = 2,
                 num_const_registers: int = 3,
                 num_calc_registers: int = 5,
                 func_set: List[int] = None):
        """
        初始化LGP算法参数

        Args:
            pop_size: 种群大小
            min_length: 程序最小长度
            max_length: 程序最大长度
            generations: 演化代数
            tournament_size: 锦标赛选择大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            num_input_registers: 输入寄存器数量
            num_const_registers: 常数寄存器数量
            num_calc_registers: 计算寄存器数量
            func_set: 函数集
        """
        self.pop_size = pop_size
        self.min_length = min_length
        self.max_length = max_length
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # 寄存器配置
        self.num_input_registers = num_input_registers
        self.num_const_registers = num_const_registers
        self.num_calc_registers = num_calc_registers
        self.total_registers = num_input_registers + num_const_registers + num_calc_registers

        # 默认函数集：加减乘除
        self.func_set = func_set if func_set else [ADD, SUB, MUL, DIV]

        # 常数寄存器值（可配置）
        self.const_registers = [0.0, 1.0, 2.0]  # 包含0和1，更容易构造简单表达式

        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.fitness_history = []

    def create_random_instruction(self) -> Tuple:
        """
        创建随机指令

        Returns:
            随机指令 (func_id, out_reg, in_reg1, in_reg2)
        """
        func_id = random.choice(self.func_set)
        # 输出寄存器应该是计算寄存器（避免覆盖输入）
        out_reg = random.randint(self.num_input_registers + self.num_const_registers,
                                self.total_registers - 1)
        in_reg1 = random.randint(0, self.total_registers - 1)
        in_reg2 = random.randint(0, self.total_registers - 1)

        return (func_id, out_reg, in_reg1, in_reg2)

    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.pop_size):
            length = random.randint(self.min_length, self.max_length)
            instructions = [self.create_random_instruction() for _ in range(length)]
            self.population.append(LGPProgram(instructions))

    def mse_fitness(self, program: LGPProgram, dataset: List[Tuple]) -> float:
        """
        计算均方差适应度

        Args:
            program: LGP程序
            dataset: 数据集，每个元素为 (inputs, target)

        Returns:
            均方差值
        """
        errors = []
        for inputs, target in dataset:
            output = program.execute(inputs, self.const_registers, self.num_calc_registers)
            # 防止输出过大或过小
            if abs(output) > 1e10 or math.isnan(output) or math.isinf(output):
                output = 0.0
            errors.append((output - target) ** 2)

        mse = sum(errors) / len(errors)

        # 添加长度惩罚，鼓励简洁的程序
        length_penalty = 0.001 * len(program.instructions)

        return mse + length_penalty

    def evaluate_population(self, dataset: List[Tuple]):
        """评估整个种群的适应度"""
        for individual in self.population:
            individual.fitness = self.mse_fitness(individual, dataset)

            # 更新最佳个体
            if individual.fitness < self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = individual.copy()

    def tournament_selection(self) -> Tuple[LGPProgram, LGPProgram]:
        """
        锦标赛选择

        Returns:
            选择出的两个父代个体
        """
        selected = random.sample(self.population, self.tournament_size)
        selected.sort(key=lambda ind: ind.fitness)
        return selected[0], selected[1]

    def crossover(self, parent1: LGPProgram, parent2: LGPProgram) -> Tuple[LGPProgram, LGPProgram]:
        """
        双点交叉

        Args:
            parent1, parent2: 父代个体

        Returns:
            两个子代个体
        """
        # 以一定概率不进行交叉
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # 确保有足够的长度进行交叉
        if len(parent1.instructions) < 2 or len(parent2.instructions) < 2:
            return parent1.copy(), parent2.copy()

        # 选择交叉点
        min_len = min(len(parent1.instructions), len(parent2.instructions))
        point1 = random.randint(0, min_len - 1)
        point2 = random.randint(point1, min_len)

        # 执行交叉
        child1_instructions = (parent1.instructions[:point1] +
                              parent2.instructions[point1:point2] +
                              parent1.instructions[point2:])

        child2_instructions = (parent2.instructions[:point1] +
                              parent1.instructions[point1:point2] +
                              parent2.instructions[point2:])

        # 确保子代长度在合理范围内
        child1_instructions = child1_instructions[:self.max_length]
        child2_instructions = child2_instructions[:self.max_length]

        # 确保最小长度
        if len(child1_instructions) < self.min_length:
            child1_instructions += [self.create_random_instruction()
                                   for _ in range(self.min_length - len(child1_instructions))]
        if len(child2_instructions) < self.min_length:
            child2_instructions += [self.create_random_instruction()
                                   for _ in range(self.min_length - len(child2_instructions))]

        return LGPProgram(child1_instructions), LGPProgram(child2_instructions)

    def mutate(self, program: LGPProgram):
        """变异操作"""
        new_instructions = program.instructions.copy()

        for i in range(len(new_instructions)):
            if random.random() < self.mutation_rate:
                # 随机变异一条指令
                new_instructions[i] = self.create_random_instruction()

        # 以一定概率插入新指令
        if random.random() < self.mutation_rate and len(new_instructions) < self.max_length:
            insert_pos = random.randint(0, len(new_instructions))
            new_instructions.insert(insert_pos, self.create_random_instruction())

        # 以一定概率删除一条指令
        if random.random() < self.mutation_rate and len(new_instructions) > self.min_length:
            delete_pos = random.randint(0, len(new_instructions) - 1)
            new_instructions.pop(delete_pos)

        program.instructions = new_instructions

    def replace_worst(self, new_individuals: List[LGPProgram]):
        """替换种群中最差的个体"""
        # 合并新旧个体
        all_individuals = self.population + new_individuals

        # 按适应度排序，选择最好的pop_size个
        all_individuals.sort(key=lambda ind: ind.fitness)
        self.population = all_individuals[:self.pop_size]

    def evolve(self, dataset: List[Tuple], verbose: bool = True):
        """
        执行演化过程

        Args:
            dataset: 训练数据集
            verbose: 是否打印进度信息
        """
        # 初始化种群
        self.initialize_population()
        self.evaluate_population(dataset)

        if verbose:
            print(f"初始最佳适应度: {self.best_fitness:.6f}")

        # 演化循环
        for gen in range(self.generations):
            # 选择父代
            parent1, parent2 = self.tournament_selection()

            # 交叉产生子代
            child1, child2 = self.crossover(parent1, parent2)

            # 变异
            self.mutate(child1)
            self.mutate(child2)

            # 评估子代
            child1.fitness = self.mse_fitness(child1, dataset)
            child2.fitness = self.mse_fitness(child2, dataset)

            # 替换最差个体
            self.replace_worst([child1, child2])

            # 重新评估种群并更新最佳个体
            self.evaluate_population(dataset)
            self.fitness_history.append(self.best_fitness)

            if verbose and (gen + 1) % 20 == 0:
                print(f"代 {gen+1}: 最佳适应度 = {self.best_fitness:.6f}")

        if verbose:
            print(f"最终最佳适应度: {self.best_fitness:.6f}")
            print("\n最佳个体程序:")
            print(self.best_individual)

    def predict(self, inputs: List[float]) -> float:
        """使用最佳个体进行预测"""
        if self.best_individual is None:
            raise ValueError("尚未训练模型，请先调用evolve方法")

        return self.best_individual.execute(inputs, self.const_registers, self.num_calc_registers)

def create_symbolic_regression_dataset(func: Callable, num_samples: int = 100,
                                      x_range: Tuple[float, float] = (-5, 5)) -> List[Tuple]:
    """
    创建符号回归数据集

    Args:
        func: 目标函数
        num_samples: 样本数量
        x_range: x的取值范围

    Returns:
        数据集列表，每个元素为 (inputs, target)
    """
    dataset = []
    for _ in range(num_samples):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(x_range[0], x_range[1])
        target = func(x, y)
        dataset.append(([x, y], target))

    return dataset

# 示例目标函数
def target_function1(x: float, y: float) -> float:
    """示例目标函数: f(x,y) = x + y (简单线性函数)"""
    return x + y

def target_function2(x: float, y: float) -> float:
    """示例目标函数: f(x,y) = x * y"""
    return x * y

def target_function3(x: float, y: float) -> float:
    """示例目标函数: f(x,y) = x^2 + y^2"""
    return x**2 + y**2

def target_function4(x: float, y: float) -> float:
    """示例目标函数: f(x,y) = sin(x) + cos(y)"""
    return math.sin(x) + math.cos(y)