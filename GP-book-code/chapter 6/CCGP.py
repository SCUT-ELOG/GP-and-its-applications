# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 6 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import random
from typing import List, Callable, Tuple
from HBWS import HBWS_schedule
from RandomExpressionTreeGeneration import random_expression, Node, TERMINAL_SET_TSR, TERMINAL_SET_RSR

# 求值表达式树
def evaluate_tree(node: Node, wf, task_idx: int, res_idx: int = None) -> float:
    if not node.children:
        # 终端节点：返回基于上下文的值
        if node.value == 'exec_time':
            return wf.exec_time[task_idx][res_idx] if res_idx is not None else 10
        elif node.value == 'EST':
            return 0  # 简化
        elif node.value in TERMINAL_SET_TSR or node.value in TERMINAL_SET_RSR:
            return 1.0  # 简化
        else:
            # 尝试作为数字处理
            try:
                return float(node.value)
            except ValueError:
                return 0
    # 函数节点：递归求值
    vals = [evaluate_tree(c, wf, task_idx, res_idx) for c in node.children]
    if node.value == '+': return vals[0] + vals[1]
    if node.value == '-': return vals[0] - vals[1]
    if node.value == '*': return vals[0] * vals[1]
    if node.value == 'DIV': return vals[0] / vals[1] if vals[1] != 0 else 1
    if node.value == 'MIN': return min(vals[0], vals[1])
    if node.value == 'MAX': return max(vals[0], vals[1])
    if node.value == 'SQRT': return vals[0] ** 0.5 if vals[0] >= 0 else 0
    if node.value == 'LOG':
        import math
        return math.log(vals[0]) if vals[0] > 0 else 0
    return 0

# 个体定义
class Individual:
    def __init__(self, tree=None):
        self.tree = tree
        self.fitness = float('inf')     # 初始适应度

# 初始化种群
def initialize_population(pop_size: int, is_tsr: bool = True) -> List[Individual]:
    return [Individual(tree=random_expression(max_depth=3, is_tsr=is_tsr))
            for _ in range(pop_size)]

# 适应度评估函数：结合另一子种群的最佳个体，评估个体性能
def evaluate_individual(ind: Individual, fixed: Individual,
                        workflow_set, scheduler, is_tsr: bool) -> float:
    total_makespan = 0.0
    for wf in workflow_set:
        # 将树转换为可调用函数
        if is_tsr:
            TSR = lambda t: evaluate_tree(ind.tree, wf, t)
            RSR = lambda t, r: evaluate_tree(fixed.tree, wf, t, r)
        else:
            TSR = lambda t: evaluate_tree(fixed.tree, wf, t)
            RSR = lambda t, r: evaluate_tree(ind.tree, wf, t, r)

        _, _, makespan = scheduler(wf, TSR, RSR)
        total_makespan += makespan
    return total_makespan / len(workflow_set)

# 差分进化 + frequency-based assignment
def mutate_tree(base, tree1, tree2):
    # 深拷贝树
    def copy_tree(node):
        if not node.children:
            return Node(node.value)
        return Node(node.value, [copy_tree(c) for c in node.children])

    # 差分变异策略
    r = random.random()
    if r < 0.3:
        return copy_tree(tree1)
    elif r < 0.6:
        return copy_tree(tree2)
    return copy_tree(base)


def gep_differential_mutation(base: Individual, r1: Individual, r2: Individual) -> Individual:
    new_tree = mutate_tree(base.tree, r1.tree, r2.tree)
    return Individual(tree=new_tree)

# 主调度器 CCGP
def CCGP(workflow_set, scheduler: Callable = None, pop_size=30, generations=50) -> Tuple[Individual, Individual]:
    if scheduler is None:
        scheduler = HBWS_schedule

    P_TSR = initialize_population(pop_size, is_tsr=True)
    P_RSR = initialize_population(pop_size, is_tsr=False)

    # 初次评估
    for ind in P_TSR:
        ind.fitness = evaluate_individual(ind, random.choice(P_RSR), workflow_set, scheduler, is_tsr=True)
    for ind in P_RSR:
        ind.fitness = evaluate_individual(ind, random.choice(P_TSR), workflow_set, scheduler, is_tsr=False)

    g_TSR = min(P_TSR, key=lambda x: x.fitness)
    g_RSR = min(P_RSR, key=lambda x: x.fitness)

    for gen in range(generations):
        # TSR 子种群进化
        new_TSR = []
        for i in range(pop_size):
            candidates = [j for j in range(pop_size) if j != i]
            if len(candidates) >= 2:
                r1, r2 = random.sample(candidates, 2)
            else:
                # 边界情况：种群太小
                r1, r2 = candidates[0], candidates[0]
            trial = gep_differential_mutation(P_TSR[i], P_TSR[r1], P_TSR[r2])
            trial.fitness = evaluate_individual(trial, g_RSR, workflow_set, scheduler, is_tsr=True)
            new_TSR.append(trial if trial.fitness < P_TSR[i].fitness else P_TSR[i])
        P_TSR = new_TSR
        g_TSR = min(P_TSR, key=lambda x: x.fitness)

        # RSR 子种群进化
        new_RSR = []
        for i in range(pop_size):
            candidates = [j for j in range(pop_size) if j != i]
            if len(candidates) >= 2:
                r1, r2 = random.sample(candidates, 2)
            else:
                # 边界情况：种群太小
                r1, r2 = candidates[0], candidates[0]
            trial = gep_differential_mutation(P_RSR[i], P_RSR[r1], P_RSR[r2])
            trial.fitness = evaluate_individual(trial, g_TSR, workflow_set, scheduler, is_tsr=False)
            new_RSR.append(trial if trial.fitness < P_RSR[i].fitness else P_RSR[i])
        P_RSR = new_RSR
        g_RSR = min(P_RSR, key=lambda x: x.fitness)

    return g_TSR, g_RSR
