# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 2 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
基于 DEAP 的标准遗传编程符号回归引擎

提供 run_gp() 函数作为主入口，接受数据 (X, y) 和超参数，
返回最佳表达式、编译函数、fitness 历史等结果字典。
"""

import operator
import random
import warnings
from functools import partial

import numpy as np
from deap import base, creator, gp, tools


# ============================================================
# 保护性运算 — 支持 numpy 数组批量计算，不使用 try-except
# ============================================================

def protected_div(a, b):
    """保护性除法：b=0 时返回 1.0"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.where(b != 0, np.divide(a, b), 1.0)


def protected_log(x):
    """保护性对数：避免 log(0)"""
    return np.log(np.abs(x) + 1e-10)


def protected_sqrt(x):
    """保护性平方根：对负数取绝对值"""
    return np.sqrt(np.abs(x))


def protected_exp(x):
    """保护性指数：限制指数范围防止溢出"""
    return np.exp(np.clip(x, -100, 100))


# 函数名到实现的映射
FUNCTION_MAP = {
    "add": (operator.add, 2),
    "sub": (operator.sub, 2),
    "mul": (operator.mul, 2),
    "div": (protected_div, 2),
    "sin": (np.sin, 1),
    "cos": (np.cos, 1),
    "log": (protected_log, 1),
    "exp": (protected_exp, 1),
    "sqrt": (protected_sqrt, 1),
}

DEFAULT_FUNCTION_SET = list(FUNCTION_MAP.keys())

def _ensure_creator():
    """确保 DEAP creator 类已注册（幂等，可直接用 hasattr 判断）"""
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree,
                       fitness=creator.FitnessMin)


# ============================================================
# PrimitiveSet 构建
# ============================================================

def make_pset(n_variables, function_set=None):
    """
    构建 DEAP GP 原语集

    Args:
        n_variables: 输入变量数量
        function_set: 使用的函数名列表，默认使用全部

    Returns:
        gp.PrimitiveSet
    """
    pset = gp.PrimitiveSet("MAIN", n_variables)

    # 重命名变量：ARG0->x0, ARG1->x1, ...
    rename_map = {f"ARG{i}": f"x{i}" for i in range(n_variables)}
    pset.renameArguments(**rename_map)

    # 注册函数
    funcs = function_set if function_set else DEFAULT_FUNCTION_SET
    for name in funcs:
        func, arity = FUNCTION_MAP[name]
        pset.addPrimitive(func, arity, name=name)

    # 短暂常数：使用 functools.partial 避免序列化警告
    pset.addEphemeralConstant("ephemeral", partial(random.uniform, -1, 1))

    return pset


# ============================================================
# 适应度评估
# ============================================================

def _make_evaluator(X, y, pset, complexity_weight):
    """
    创建适应度评估函数（闭包捕获数据和参数）

    Args:
        X: 输入数据 (n_samples, n_features)
        y: 目标值 (n_samples,)
        pset: 原语集
        complexity_weight: 复杂度惩罚权重

    Returns:
        评估函数，签名: evaluate(individual) -> (fitness,)
    """
    # 预计算转置，避免每次评估重复计算
    XT = X.T
    # 编译缓存：相同表达式不重复调用 gp.compile (含 eval)
    _compile_cache = {}

    def evaluate(individual):
        key = str(individual)
        func = _compile_cache.get(key)
        if func is None:
            func = gp.compile(expr=individual, pset=pset)
            _compile_cache[key] = func

        # 批量评估：func 签名为 func(x0, x1, ...)
        predictions = func(*XT)

        # 处理非数组返回值
        if not isinstance(predictions, np.ndarray):
            predictions = np.full(len(y), predictions)

        # 检查 NaN/Inf
        if not np.all(np.isfinite(predictions)):
            return (np.inf,)

        rmse = np.sqrt(np.mean((predictions - y) ** 2))
        complexity = len(individual)
        fitness = rmse + complexity_weight * complexity

        return (fitness,)

    return evaluate


# ============================================================
# 自定义进化循环
# ============================================================

def _evolve(toolbox, population, cxpb, mutpb, ngen, elite_size,
            patience, verbose=True):
    """
    自定义进化循环：精英保留 + 早停

    Args:
        toolbox: DEAP toolbox
        population: 初始种群
        cxpb: 交叉概率
        mutpb: 变异概率
        ngen: 最大代数
        elite_size: 精英保留数量
        patience: 早停耐心值（代数）
        verbose: 是否打印进度

    Returns:
        (population, logbook, halloffame)
    """
    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # 评估初始种群
    invalid = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit

    halloffame.update(population)
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(invalid), **record)
    if verbose:
        print(f"Gen 0: min={record['min']:.6f} avg={record['avg']:.6f}")

    best_fitness = halloffame[0].fitness.values[0]
    stagnation = 0

    for gen in range(1, ngen + 1):
        # 精英保留：直接复制 top-k
        elites = tools.selBest(population, elite_size)

        # 选择 + 交叉 + 变异填充剩余位置
        offspring = toolbox.select(population, len(population) - elite_size)
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(
                    offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # 变异
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # 评估无效个体
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # 新种群 = 精英 + 后代
        population[:] = elites + offspring

        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid), **record)

        if verbose and gen % 10 == 0:
            print(f"Gen {gen}: min={record['min']:.6f} avg={record['avg']:.6f}")

        # 早停检查
        current_best = halloffame[0].fitness.values[0]
        if current_best < best_fitness:
            best_fitness = current_best
            stagnation = 0
        else:
            stagnation += 1

        if patience > 0 and stagnation >= patience:
            if verbose:
                print(f"Early stopping at gen {gen}")
            break

    return population, logbook, halloffame


# ============================================================
# 主入口
# ============================================================

def run_gp(X, y, pop_size=500, generations=40, cxpb=0.7, mutpb=0.1,
           elite_size=5, tournament_size=7, max_depth=17,
           init_depth=(1, 2), complexity_weight=0.01,
           function_set=None, patience=20, seed=None, verbose=True):
    """
    运行标准遗传编程符号回归

    Args:
        X: 输入数据，shape (n_samples, n_features)
        y: 目标值，shape (n_samples,)
        pop_size: 种群大小
        generations: 最大进化代数
        cxpb: 交叉概率
        mutpb: 变异概率
        elite_size: 精英保留数量
        tournament_size: 锦标赛选择大小
        max_depth: 树最大深度（staticLimit）
        init_depth: 初始化树深度范围 (min, max)
        complexity_weight: 复杂度惩罚权重
        function_set: 函数集，默认使用全部
        patience: 早停耐心值，0 表示不启用
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        dict 包含:
            best_expression: 最佳表达式字符串
            best_func: 编译后的可调用函数
            best_fitness: 最佳适应度值
            best_complexity: 最佳个体节点数
            history: fitness 历史 {min, avg, max}
            generations_used: 实际使用代数
    """
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_features = X.shape[1]

    # 确保 creator 类已创建
    _ensure_creator()

    # 构建原语集
    pset = make_pset(n_features, function_set)

    # 构建 toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                     min_=init_depth[0], max_=init_depth[1])
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)

    # 适应度评估
    evaluate = _make_evaluator(X, y, pset, complexity_weight)
    toolbox.register("evaluate", evaluate)

    # 遗传操作
    toolbox.register("select", tools.selTournament,
                     tournsize=tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, pset=pset,
                     min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform,
                     expr=toolbox.expr_mut, pset=pset)

    # 树深度限制（防止 bloat）
    toolbox.decorate("mate",
                     gp.staticLimit(key=operator.attrgetter("height"),
                                    max_value=max_depth))
    toolbox.decorate("mutate",
                     gp.staticLimit(key=operator.attrgetter("height"),
                                    max_value=max_depth))

    # 初始化种群
    population = toolbox.population(n=pop_size)

    # 运行进化
    population, logbook, halloffame = _evolve(
        toolbox, population, cxpb, mutpb, generations,
        elite_size, patience, verbose
    )

    # 编译最佳个体，封装为统一签名：best_func(X) → y
    best = halloffame[0]
    raw_func = gp.compile(expr=best, pset=pset)

    def best_func(X):
        """预测函数：输入 (n_samples, n_features) 矩阵，输出 (n_samples,) 向量"""
        predictions = raw_func(*X.T)
        if not isinstance(predictions, np.ndarray):
            return np.full(X.shape[0], predictions)
        return predictions

    # 提取历史
    history = {
        "min": logbook.select("min"),
        "avg": logbook.select("avg"),
        "max": logbook.select("max"),
    }
    gens_used = len(logbook) - 1  # 排除 gen 0

    return {
        "best_expression": str(best),
        "best_func": best_func,
        "best_fitness": best.fitness.values[0],
        "best_complexity": len(best),
        "history": history,
        "generations_used": gens_used,
        "logbook": logbook,
    }
