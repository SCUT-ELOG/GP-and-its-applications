# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 2 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import numpy as np
from anytree import Node, RenderTree
import matplotlib.pyplot as plt
from random import randint
from warnings import warn

import numpy as np
from anytree import Node, RenderTree
import matplotlib.pyplot as plt
from random import randint
from warnings import warn


class Chromosome:
    """
    基因表达式编程(GEP)染色体类
    实现了GEP算法的核心功能，包括染色体表示、表达式树构建、评估和适应度计算
    """
    # 类变量 - 所有染色体实例共享的配置
    functions = dict()  # 函数集字典，格式：{'函数名': {'f': 函数对象, 'args': 参数个数}}
    terminals = list()  # 终止符列表，如变量名和常数
    constants = dict()  # 常数字典，预定义的常数映射
    ephemeral_random_constants_range = (-1, 1)  # 临时随机常数的生成范围
    linking_function = None  # 多基因链接函数

    # 染色体结构参数
    num_genes = 3  # 基因数量
    head_length = 6  # 基因头部长度
    length = 39  # 染色体总长度（每个基因的长度）

    # 适应度计算相关
    fitness_cases = []  # 适应度案例列表，格式：[(输入字典, 目标输出), ...]
    max_fitness = None  # 最大适应度值

    def __init__(self, genes: list):
        """
        初始化染色体

        """
        # 验证类变量是否已正确设置
        if not Chromosome.functions:
            raise ValueError("Chromosome class has no functions associated with it.")
        if len(Chromosome.terminals) == 0:
            raise ValueError("Chromosome class has no terminals associated with it.")
        if Chromosome.length is None:
            raise ValueError("Chromosome class has no length defined.")
        if Chromosome.head_length is None:
            raise ValueError("Chromosome class has no head length defined.")
        if Chromosome.linking_function is None and len(genes) > 1:
            raise ValueError("Multigenic chromosome defined with no linking function.")
        if len(genes) != Chromosome.num_genes:
            raise ValueError("Number of genes does not match excpected value in class level variable.")
        if "?" in Chromosome.terminals and Chromosome.ephemeral_random_constants_range is None:
            raise ValueError("Must define ephemeral random constants range if using ephemeral random constants.")

        # 初始化实例变量
        self.genes = genes  # 存储基因字符串列表
        self.trees = []  # 缓存构建的表达式树
        self._values_ = {}  # 缓存评估结果，提高性能
        self._fitness_ = None  # 缓存适应度值
        # 生成临时随机常数，用于替换基因中的"?"符号
        self.ephemeral_random_constants = list(np.random.uniform(
            *Chromosome.ephemeral_random_constants_range,
            size=Chromosome.length
        ))

    def evaluate(self, terminal_values: dict) -> float:
        """
        评估染色体在当前输入下的输出值
        Returns:
            染色体表达式的计算结果
        """
        # 使用指纹技术缓存结果，避免重复计算
        value_fingerprint = tuple(sorted(terminal_values.items()))
        if value_fingerprint in self._values_:
            return self._values_[value_fingerprint]

        # 如果表达式树未构建，则构建所有基因的表达式树
        if len(self.trees) == 0:
            self.trees = [Chromosome.build_tree(gene) for gene in self.genes]

        # 多基因染色体使用链接函数组合，单基因直接使用第一个树
        if self.num_genes > 1:
            expression_tree = Chromosome.link(*self.trees)
        else:
            expression_tree = self.trees[0]

        erc_index = 0  # 临时随机常数索引

        def inorder(start: Node) -> float:
            """
            递归中序遍历表达式树并计算值
            Returns:
                子树的计算结果
            """
            nonlocal terminal_values, erc_index
            # 如果是终止符
            if start.name in Chromosome.terminals:
                if start.name == "?":  # 临时随机常数
                    erc_index += 1
                    return self.ephemeral_random_constants[erc_index - 1]
                if start.name in Chromosome.constants:  # 预定义常数
                    return Chromosome.constants[start.name]
                # 数字或变量
                return int(start.name) if start.name.isdigit() else terminal_values[start.name]
            # 如果是函数，递归计算所有子节点
            if start.name in Chromosome.functions:
                return Chromosome.functions[start.name]["f"](
                    *[inorder(node) for node in start.children]
                )
        try:
            # 计算表达式值并缓存
            self._values_[value_fingerprint] = inorder(expression_tree)
            # 修复：使用 complex 而不是 np.complex
            if isinstance(self._values_[value_fingerprint], complex):
                raise TypeError
        # 处理除零错误和类型错误（如负数的平方根）
        except (ZeroDivisionError, TypeError):
            self._values_[value_fingerprint] = np.nan

        return self._values_[value_fingerprint]

    def fitness(self) -> float:
        """
        获取染色体的适应度值
        Returns:
            适应度值，如果未计算则返回0并警告
        """
        if self._fitness_ is not None:
            return self._fitness_
        warn("Fitness of chromosome has not been properly calculated. Returning 0.")
        return 0

    def print_tree(self) -> None:
        """打印染色体对应的表达式树结构"""
        for t in range(len(self.trees)):
            print("Tree %d" % t)
            for pre, _, node in RenderTree(self.trees[t]):
                print("\t%s%s" % (pre, node.name))
        print(self.ephemeral_random_constants)

    def plot_solution(self, objective_function, x_min: float, x_max: float,
                      avg_fitnesses: list, best_fitnesses: list, variable_name: str) -> None:
        """
        绘制符号回归结果和适应度进化曲线
        """
        if objective_function is not None:
            plt.subplots(1, 2, figsize=(16, 8))

            # 左图：发现函数 vs 目标函数
            xs = np.linspace(x_min, x_max, 100)
            plt.subplot(1, 2, 1)
            plt.title("Discovered function vs. Objective function")
            plt.plot(xs, [objective_function(x) for x in xs],
                     linewidth=2, linestyle='dashed', color='black', label="Objective")
            plt.plot(xs, [self.evaluate({variable_name: x}) for x in xs],
                     linewidth=2, color='blue', label="Discovered")
            plt.legend(loc="upper left")

            # 右图：适应度进化曲线
            plt.subplot(1, 2, 2)
            plt.title("Fitness by Generation")
            plt.plot(range(len(avg_fitnesses)), avg_fitnesses, label="Average")
            plt.plot(range(len(best_fitnesses)), best_fitnesses, label="Best")
            plt.legend(loc="upper left")
            plt.show()
        else:
            # 只绘制适应度曲线
            plt.subplots(1, 1, figsize=(8, 8))
            plt.title("Fitness by Generation")
            plt.plot(range(len(avg_fitnesses)), avg_fitnesses, label="Average")
            plt.plot(range(len(best_fitnesses)), best_fitnesses, label="Best")
            plt.legend(loc="upper left")
            plt.show()

    @staticmethod
    def build_tree(gene: str) -> Node:
        """
        从基因字符串构建表达式树
        Returns:
            表达式树的根节点
        """
        def args(f: str) -> int:
            """获取函数的参数个数"""
            return Chromosome.functions[f]["args"] if f in Chromosome.functions else 0

        def build_tree_recursive(gene_str: str, parent: Node = None) -> (Node, str):
            """
            递归构建表达式树

            Returns:
                (当前节点, 剩余的基因字符串)
            """
            if not gene_str:
                # 如果基因字符串为空，返回一个终止符节点
                terminal = Chromosome.terminals[0] if Chromosome.terminals else "0"
                return Node(terminal, parent=parent), ""

            # 获取当前符号
            current_symbol = gene_str[0]
            remaining_str = gene_str[1:]

            # 创建当前节点
            current_node = Node(current_symbol, parent=parent)

            # 如果当前符号是函数，则递归构建子节点
            if current_symbol in Chromosome.functions:
                nargs = args(current_symbol)
                for _ in range(nargs):
                    if not remaining_str:
                        # 如果剩余字符串为空，添加一个终止符节点
                        terminal = Chromosome.terminals[0] if Chromosome.terminals else "0"
                        Node(terminal, parent=current_node)
                    else:
                        child_node, remaining_str = build_tree_recursive(remaining_str, current_node)

            return current_node, remaining_str

        # 使用新的递归方法构建树
        root, _ = build_tree_recursive(gene)
        return root
    @staticmethod
    def link(*args) -> Node:
        """
        使用链接函数连接多个表达式树
        Returns:
            连接后的表达式树根节点
        """
        if Chromosome.linking_function not in Chromosome.functions:
            raise ValueError("Linking function is not defined in Chromosome.functions.")
        if not all([isinstance(arg, Node) for arg in args]):
            raise TypeError("Can only link expression trees.")

        nargs = Chromosome.functions[Chromosome.linking_function]["args"]  # 链接函数的参数个数

        def link_recursive(*args) -> Node:
            """递归链接树，处理参数数量不匹配的情况"""
            root = Node(Chromosome.linking_function)
            if len(args) == nargs:
                # 参数数量匹配，直接连接
                for tree in args:
                    tree.parent = root
                return root
            else:
                # 参数数量不匹配，递归链接
                return link_recursive(link_recursive(*args[:nargs]), *args[nargs:])

        return link_recursive(*args)

    @staticmethod
    def absolute_fitness(M: float, *args) -> np.ndarray:

        fitnesses = []
        for chromosome in args:
            if chromosome._fitness_ is not None:  # 使用缓存值
                fitnesses.append(chromosome._fitness_)
            else:
                fitness = 0
                for j in range(len(Chromosome.fitness_cases)):
                    C_ij = chromosome.evaluate(Chromosome.fitness_cases[j][0])  # 预测值
                    T_j = Chromosome.fitness_cases[j][1]  # 目标值

                    # 处理无效值（复数、NaN、无穷大）
                    if (isinstance(C_ij, complex) or np.isnan(C_ij) or
                            np.isinf(C_ij) or np.isneginf(C_ij)):
                        fitness = 0
                        break
                    fitness += M - abs(C_ij - T_j)
                chromosome._fitness_ = fitness
                fitnesses.append(fitness)
        return np.asarray(fitnesses)

    @staticmethod
    def relative_fitness(M: float, *args) -> np.ndarray:

        fitnesses = []
        for chromosome in args:
            if chromosome._fitness_ is not None:
                fitnesses.append(chromosome._fitness_)
            else:
                fitness = 0
                for j in range(len(Chromosome.fitness_cases)):
                    C_ij = chromosome.evaluate(Chromosome.fitness_cases[j][0])
                    T_j = Chromosome.fitness_cases[j][1]
                    fitness += M - 100 * abs(C_ij / T_j - 1)
                chromosome._fitness_ = fitness
                fitnesses.append(fitness)
        return np.asarray(fitnesses)

    @staticmethod
    def inv_squared_error(*args) -> np.ndarray:

        fitnesses = []
        for chromosome in args:
            if chromosome._fitness_ is not None:
                fitnesses.append(chromosome._fitness_)
            else:
                fitness = 0
                for j in range(len(Chromosome.fitness_cases)):
                    C_ij = chromosome.evaluate(Chromosome.fitness_cases[j][0])
                    T_j = Chromosome.fitness_cases[j][1]

                    # 处理无效值
                    # 修复：使用 complex 而不是 np.complex
                    if (isinstance(C_ij, complex) or np.isnan(C_ij) or
                            np.isinf(C_ij) or np.isneginf(C_ij)):
                        fitness = np.inf
                        break
                    fitness += (C_ij - T_j) ** 2
                chromosome._fitness_ = 1.0 / (1 + fitness)
                fitnesses.append(chromosome._fitness_)
        return np.asarray(fitnesses)

    @staticmethod
    def centralized_inv_squared_error(center: float, dimension: str, *args) -> np.ndarray:
        fitnesses = []
        for chromosome in args:
            if chromosome._fitness_ is not None:
                fitnesses.append(chromosome._fitness_)
            else:
                fitness = 0
                for j in range(len(Chromosome.fitness_cases)):
                    C_ij = chromosome.evaluate(Chromosome.fitness_cases[j][0])
                    T_j = Chromosome.fitness_cases[j][1]

                    # 修复：使用 complex 而不是 np.complex
                    if (isinstance(C_ij, complex) or np.isnan(C_ij) or
                            np.isinf(C_ij) or np.isneginf(C_ij)):
                        fitness = np.inf
                        break
                    # 使用到中心点的距离作为误差的指数权重
                    distance_weight = abs(Chromosome.fitness_cases[j][0][dimension] - center)
                    # 避免除零错误
                    if distance_weight == 0:
                        distance_weight = 1e-10
                    fitness += abs(C_ij - T_j) ** (1 / distance_weight)
                chromosome._fitness_ = 1.0 / (1 + fitness)
                fitnesses.append(chromosome._fitness_)
        return np.asarray(fitnesses)

    @staticmethod
    def generate_random_gene() -> str:
        # 头部可以包含函数和终止符
        possible_chars = list(Chromosome.functions.keys()) + Chromosome.terminals
        # 头部随机生成
        head = "".join([possible_chars[randint(0, len(possible_chars) - 1)]
                        for _ in range(Chromosome.head_length)])
        # 尾部只能包含终止符
        tail = "".join([Chromosome.terminals[randint(0, len(Chromosome.terminals) - 1)]
                        for _ in range(Chromosome.length - Chromosome.head_length)])
        return head + tail

    @staticmethod
    def generate_random_individual() -> 'Chromosome':

        return Chromosome([Chromosome.generate_random_gene()
                           for _ in range(Chromosome.num_genes)])