# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 4 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import numpy as np
from typing import Callable, Tuple, List
# 输入：待评估解f；原始样本集X；上下界列表bounds
# 输出：适应度值V
def cl_fem(
    f: Callable[..., float],
    X: np.ndarray,
    bounds: List[Tuple[float, float]],
    eps: float = 1e-4
) -> float:
    """
    参数:
        f : 隐函数 f(x1, x2, ..., xD)，输入为若干自变量，输出为标量值
        X : 原始样本集，形状为 (N, D)
        bounds : 每一维特征变量的上下界列表 [(L1, U1), (L2, U2), ...]
        eps : 判断某一维变量是否“起作用”的阈值
    返回值:
        适应度值 V (值越小表示模型越优)
    """
    N, D = X.shape
# 计算原样本下的方程值 f(x)
    S = np.array([f(*x) for x in X])
    # 对每一维产生扰动样本并计算差异值 m_j，公式（4-6）
    m = np.zeros(D)
    for j in range(D):
        X_perturb = X.copy()
        lower, upper = bounds[j]
        X_perturb[:, j] = np.random.uniform(lower, upper, size=N)
        Sj = np.array([f(*x) for x in X_perturb])
        m[j] = np.mean((Sj - S) ** 2)  # 公式 (4-7)
    # 判断哪些变量起作用
    active_dims = m > eps
    # 计算最终适应度
    if np.all(active_dims):
        # 所有变量都对方程结果有影响
        V = np.mean(S ** 2)  # 公式 (4-8)
    else:
        # 存在无效变量或恒等方程，返回一个极大值1010
        V = 1e10
    return V

#以下生成不同示例来反映评估过程中可能遇到的情况
# 生成样本数据，这里在y - sin(x) = 0 （x∈[-,]）曲线上均匀截取200个点作为初始样本
x_vals = np.linspace(-np.pi, np.pi, 200)
y_vals = np.sin(x_vals)
X = np.column_stack((x_vals, y_vals))
bounds = [(-np.pi, np.pi), (-1.0, 1.0)]

# 定义三个待评估的隐函数
# 示例一：完美拟合的函数
def f_true(x: float, y: float) -> float:
    return y - np.sin(x)

# 示例二：有偏差的函数
def f_wrong(x: float, y: float) -> float:
    return y - np.cos(x)

# 示例三：无意义的函数
def f_insignificance(x: float, y: float) -> float:
    return x**2 + 1

for f, name in [(f_true, "完美拟合函数 f_true"),
                (f_wrong, "有偏差函数 f_wrong"),
                (f_insignificance, "无意义函数 f_insignificance")]:
    score = cl_fem(f, X, bounds)
    print(f"{name} 的适应度 V = {score:.4e}")