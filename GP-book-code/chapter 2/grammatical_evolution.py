# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 2 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import random
from deap import base, creator, tools

# 1. 使用Python字典定义BNF文法，严格对应 "图 2-18"
BNF_GRAMMAR = {
    "<expr>": [
        "<expr><op><expr>",       # 规则 (1.0)
        "(<expr><op><expr>)",     # 规则 (1.1)
        "<pre_op>(<expr>)",      # 规则 (1.2)
        "<var>",                 # 规则 (1.3)
    ],
    "<op>": [
        "+",                     # 规则 (2.0)
        "-",                     # 规则 (2.1)
        "*",                     # 规则 (2.2)
        "/",                     # 规则 (2.3)
    ],
    "<pre_op>": [
        "sin",                   # 规则 (3.0)
        "log",                   # 规则 (3.1)
        "cos",                   # 规则 (3.2)
    ],
    "<var>": [
        "x",                     # 规则 (4.0)
        "1.0",                   # 规则 (4.1)
    ]
}

# 2. 初始化DEAP框架
# 创建一个名为 "FitnessMin" 的适应度类，目标是最小化该值
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# 创建名为 "Individual" 的个体类，其结构为Python列表，并拥有 FitnessMin 属性
creator.create("Individual", list, fitness=creator.FitnessMin)

# 3. 创建一个工具箱(Toolbox)用于生成基因和个体
toolbox = base.Toolbox()

# 定义单个基因（密码子）的生成规则：一个0到255的随机整数
CODON_MAX_VALUE = 255
toolbox.register("codon", random.randint, 0, CODON_MAX_VALUE)

# 定义整个个体（染色体）的生成规则：由N个密码子组成的列表
GENOME_LENGTH = 80  # 染色体长度
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.codon, n=GENOME_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def grammatical_map(individual, grammar):
    """
    根据给定的整数个体（基因型）和BNF文法，执行映射过程。
    """
    # 步骤1: 从起始符S开始
    current_expression = "<expr>" 
    codon_index = 0
    max_derivations = 200 # 设置一个推导上限，防止无限递归
    derivations = 0

    # 循环直到表达式中没有非终结符为止
    while "<" in current_expression and derivations < max_derivations:
        # 步骤1 (循环内): 选取左起第一个非终结元素
        start = current_expression.find("<")
        end = current_expression.find(">")
        symbol_to_replace = current_expression[start:end+1]

        # 步骤3: 从个体中取得密码子
        codon = individual[codon_index]
        
        # 获取该非终结符的所有可用产生式规则
        rules = grammar[symbol_to_replace]
        # 使用模运算决定选择哪条规则
        rule_index = codon % len(rules)
        selected_rule = rules[rule_index]
        
        # 步骤2: 将非终结符替换为选中的规则
        current_expression = current_expression.replace(symbol_to_replace, selected_rule, 1)

        # 移动到下一个密码子
        codon_index += 1
        # 步骤4: 如果密码子用完，则从头开始 (Wrap)
        if codon_index >= len(individual):
            codon_index = 0
        
        derivations += 1

    # 如果推导结束但仍有非终结符，说明映射失败
    if "<" in current_expression:
        return None
        
    return current_expression

# --- 使用文本中的例子进行验证 ---
chromosome_example = [220, 122, 22, 87, 160, 12, 7, 9, 53] # 使用了一个能产生log(x)+1.0的染色体
phenotype = grammatical_map(chromosome_example, BNF_GRAMMAR)

print(f"染色体: {chromosome_example}")
print(f"映射结果 (表现型): {phenotype}")
# 预期输出: log(x)+1.0 (根据我们手动构造的染色体)


# (接续之前的 toolbox 对象)
# 注册交叉算子：采用简单的单点交叉 (cxOnePoint)
# 常用概率(cxpb): 0.7 - 0.9
toolbox.register("mate", tools.cxOnePoint)

# 注册变异算子：采用均匀整数突变 (mutUniformInt)
# 常用个体突变概率(mutpb): 0.1 - 0.2
# 常用基因内部突变概率(indpb): 0.05 (表示每个基因有5%的概率突变)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=CODON_MAX_VALUE, indpb=0.05)

# 注册选择算子：采用锦标赛选择，每一轮从3个个体中选出最优的一个
# 常用锦标赛大小(tournsize): 3 或 5
toolbox.register("select", tools.selTournament, tournsize=3)


def map_and_count_codons(individual, grammar):
    current_expression = "<expr>"
    codon_index = 0
    max_derivations = 200
    derivations = 0
    used_codons = 0  # 记录实际使用的密码子数量

    while "<" in current_expression and derivations < max_derivations:
        start = current_expression.find("<")
        end = current_expression.find(">")

        if end == -1:
            break

        symbol_to_replace = current_expression[start:end + 1]
        rules = grammar.get(symbol_to_replace)
        if not rules:
            break

        codon = individual[codon_index]
        rule_index = codon % len(rules)
        selected_rule = rules[rule_index]

        current_expression = current_expression.replace(
            symbol_to_replace, selected_rule, 1)

        codon_index += 1
        used_codons += 1  # 每次使用密码子都计数

        if codon_index >= len(individual):
            codon_index = 0

        derivations += 1

    if "<" in current_expression:
        return None, used_codons

    return current_expression, used_codons


# --- 1. 复制 (Duplicate) 算子的实现 ---
def duplicate_operator(individual, dup_prob, dup_rate):

    # 参数验证
    if not (0 <= dup_prob <= 1):
        raise ValueError("dup_prob必须在0-1之间")
    if not (0 <= dup_rate <= 1):
        raise ValueError("dup_rate必须在0-1之间")

    if random.random() < dup_prob:
        num_to_duplicate = max(1, int(len(individual) * dup_rate))

        # 确保不会选择超出范围的片段
        if num_to_duplicate >= len(individual):
            num_to_duplicate = len(individual) // 2

        start_index = random.randint(0, len(individual) - num_to_duplicate)
        segment = individual[start_index: start_index + num_to_duplicate]

        # 将片段追加到染色体末尾
        individual.extend(segment)

    return individual,  # DEAP算子通常返回一个元组


# --- 2. 剪切 (Prune) 算子的实现 ---
def prune_operator(individual, prune_prob, grammar):
    # 参数验证
    if not (0 <= prune_prob <= 1):
        raise ValueError("prune_prob必须在0-1之间")

    if random.random() < prune_prob:
        try:
            # 映射个体，并找出实际使用了多少基因
            _, used_codons = map_and_count_codons(individual, grammar)

            # 只有当实际使用的密码子数量小于染色体长度时才进行剪切
            # 并且确保至少保留一个密码子
            if 0 < used_codons < len(individual):
                del individual[used_codons:]

        except Exception as e:
            # 如果映射过程中出现错误，不进行剪切操作
            print(f"剪切操作中映射失败: {e}")

    return individual,


# 注册自定义算子
# 注意: 我们使用lambda将额外参数（概率、文法等）固定下来
toolbox.register("duplicate", duplicate_operator, dup_prob=0.05, dup_rate=0.1)
toolbox.register("prune", prune_operator, prune_prob=0.08, grammar=BNF_GRAMMAR)

# --- 操作示例 ---
# 创建一个个体
test_ind = toolbox.individual()
print(f"\n--- 复制与剪切示例 ---")
print(f"原始个体 (长度 {len(test_ind)}): {test_ind[:10]}...")

# 执行复制操作
toolbox.duplicate(test_ind)
print(f"复制后个体 (长度 {len(test_ind)}): {test_ind[:10]}...")

# 执行剪切操作
toolbox.prune(test_ind)
print(f"剪切后个体 (长度 {len(test_ind)}): {test_ind[:10]}...")