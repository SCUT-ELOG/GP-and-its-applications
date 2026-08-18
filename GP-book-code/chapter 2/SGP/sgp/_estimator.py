# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 2 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
sklearn 风格的 GP 符号回归 Estimator

提供 fit / predict / score 接口，兼容 sklearn BaseEstimator。
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from sgp._gp import run_gp


class SGP(BaseEstimator, RegressorMixin):
    """
    Standard Genetic Programming for Symbolic Regression

    基于 DEAP 的标准 GP 符号回归，提供 sklearn 兼容接口。

    Parameters
    ----------
    pop_size : int
        种群大小
    generations : int
        最大进化代数
    cxpb : float
        交叉概率
    mutpb : float
        变异概率
    elite_size : int
        精英保留数量
    tournament_size : int
        锦标赛选择大小
    max_depth : int
        树最大深度
    init_depth : tuple
        初始化树深度范围 (min, max)
    complexity_weight : float
        复杂度惩罚权重
    function_set : list or None
        函数集，默认使用全部
    patience : int
        早停耐心值，0 表示不启用
    seed : int or None
        随机种子
    verbose : bool
        是否打印进度

    Attributes
    ----------
    best_expression_ : str
        最佳表达式字符串
    best_func_ : callable
        编译后的最佳函数
    best_fitness_ : float
        最佳适应度值
    best_complexity_ : int
        最佳个体节点数
    history_ : dict
        fitness 历史 {min, avg, max}
    n_generations_ : int
        实际使用代数
    n_features_in_ : int
        输入特征数
    """

    def __init__(self, pop_size=500, generations=40, cxpb=0.7, mutpb=0.1,
                 elite_size=5, tournament_size=7, max_depth=17,
                 init_depth=(1, 2), complexity_weight=0.01,
                 function_set=None, patience=20, seed=None, verbose=True):
        self.pop_size = pop_size
        self.generations = generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.init_depth = init_depth
        self.complexity_weight = complexity_weight
        self.function_set = function_set
        self.patience = patience
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        """
        运行 GP 符号回归

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            输入数据
        y : array-like, shape (n_samples,)
            目标值

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]

        result = run_gp(
            X, y,
            pop_size=self.pop_size,
            generations=self.generations,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            elite_size=self.elite_size,
            tournament_size=self.tournament_size,
            max_depth=self.max_depth,
            init_depth=self.init_depth,
            complexity_weight=self.complexity_weight,
            function_set=self.function_set,
            patience=self.patience,
            seed=self.seed,
            verbose=self.verbose,
        )

        self.best_expression_ = result["best_expression"]
        self.best_func_ = result["best_func"]
        self.best_fitness_ = result["best_fitness"]
        self.best_complexity_ = result["best_complexity"]
        self.history_ = result["history"]
        self.n_generations_ = result["generations_used"]

        return self

    def predict(self, X):
        """
        使用最佳表达式预测

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.best_func_(X)

    def plot_fitness(self, save_path="figures/fitness_evolution.svg"):
        """
        绘制 fitness 进化曲线（矢量图）

        Parameters
        ----------
        save_path : str
            保存路径，默认 figures/fitness_evolution.svg。
            支持 .svg / .pdf 等矢量格式。

        Returns
        -------
        save_path : str
            实际保存路径
        """
        check_is_fitted(self)
        import os
        import matplotlib
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        gens = range(len(self.history_["min"]))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(gens, self.history_["min"], label="Best", linewidth=1.5)
        ax.plot(gens, self.history_["avg"], label="Average", linewidth=1, alpha=0.7)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (RMSE + penalty)")
        ax.set_title(f"Fitness Evolution — {self.best_expression_}")
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path
