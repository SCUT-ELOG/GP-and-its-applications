# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 6 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

from typing import List, Dict, Set, Callable, Tuple


class Task:
    def __init__(self, index: int):
        self.index = index
        self.parents: List[int] = []           # 父任务编号列表
        self.children: List[int] = []          # 子任务编号列表

class Workflow:
    def __init__(self, task_amount: int, resource_amount: int):
        self.task_amount = task_amount
        self.resource_amount = resource_amount
        self.tasks: List[Task] = [Task(i) for i in range(task_amount)]
        self.exec_time: List[List[float]] = [
            [0.0 for _ in range(resource_amount)] for _ in range(task_amount)
        ]
        self.transfer_time: List[List[float]] = [
            [0.0 for _ in range(task_amount)] for _ in range(task_amount)
        ]

def HBWS_schedule(
    wf: Workflow,
    TSR: Callable[[int], float],
    RSR: Callable[[int, int], float]
) -> Tuple[List[int], Dict[int, int], float]:

    LET = [0.0 for _ in range(wf.resource_amount)]  # 每个资源的最早空闲时间
    AFT = {}                                 # 每个任务的完成时间
    M = {}                                   # M[t_i] = p_j：任务-资源映射
    O = []                                    # O：调度顺序（任务选择顺序）
    makespan = 0.0
    completed: Set[int] = set()
    parent_remaining = {t.index: len(t.parents) for t in wf.tasks}
    WQ = [t.index for t in wf.tasks if not t.parents]  # 初始等待队列（入口任务）

    while WQ:
        # 选择优先级最高的任务 t_i、最优资源 p_j
        t_i = max(WQ, key=lambda t: TSR(t))
        p_j = max(range(wf.resource_amount), key=lambda j: RSR(t_i, j))

        # 计算 AST
        if wf.tasks[t_i].parents:
            pred_ready = []
            for t_k in wf.tasks[t_i].parents:
                p_k = M[t_k]
                c_ki = wf.transfer_time[t_k][t_i] if p_k != p_j else 0.0
                pred_ready.append(AFT[t_k] + c_ki)
            AST = max(LET[p_j], max(pred_ready))
        else:
            AST = LET[p_j]  # 入口任务，无前驱

        # 计算 AFT
        AFT_i = AST + wf.exec_time[t_i][p_j]
        # 更新 makespan、LET、AFT、M、O
        makespan = max(makespan, AFT_i)
        LET[p_j] = AFT_i
        AFT[t_i] = AFT_i
        M[t_i] = p_j
        O.append(t_i)

        # 从等待队列中移除 t_i
        WQ.remove(t_i)
        completed.add(t_i)

        # 更新其所有子任务 t_c 的状态
        for t_c in wf.tasks[t_i].children:
            parent_remaining[t_c] -= 1
            if parent_remaining[t_c] == 0 and t_c not in WQ and t_c not in completed:
                WQ.append(t_c)


    return O, M, makespan
