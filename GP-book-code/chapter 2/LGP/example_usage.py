# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 2 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
LGP算法使用示例 - 改进版本
"""

from lgp import LGP, create_symbolic_regression_dataset, target_function1, target_function2, target_function3, target_function4
import matplotlib.pyplot as plt

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def example_simple():
    """示例1: 简单的线性函数回归"""
    print("=== LGP简单线性函数回归示例 ===")

    # 创建数据集
    dataset = create_symbolic_regression_dataset(target_function1, num_samples=50, x_range=(-5, 5))

    # 创建LGP实例
    lgp = LGP(
        pop_size=100,
        min_length=3,
        max_length=15,
        generations=200,
        tournament_size=5,
        mutation_rate=0.03,
        crossover_rate=0.8,
        num_input_registers=2,
        num_const_registers=3,
        num_calc_registers=5
    )

    # 执行演化
    lgp.evolve(dataset, verbose=True)

    # 测试预测
    print("\n=== 预测测试 ===")
    test_inputs = [2.0, 3.0]
    prediction = lgp.predict(test_inputs)
    actual = target_function1(test_inputs[0], test_inputs[1])
    print(f"输入: {test_inputs}")
    print(f"预测值: {prediction:.4f}")
    print(f"实际值: {actual:.4f}")
    print(f"误差: {abs(prediction - actual):.4f}")

    return lgp

def example_multiplication():
    """示例2: 乘法函数回归"""
    print("\n=== 乘法函数回归示例 ===")

    # 创建数据集
    dataset = create_symbolic_regression_dataset(target_function2, num_samples=100, x_range=(-3, 3))

    # 创建LGP实例
    lgp = LGP(
        pop_size=150,
        min_length=5,
        max_length=20,
        generations=300,
        tournament_size=5,
        mutation_rate=0.02,
        crossover_rate=0.8,
        num_input_registers=2,
        num_const_registers=3,
        num_calc_registers=5
    )

    # 执行演化
    lgp.evolve(dataset, verbose=True)

    # 测试预测
    print("\n=== 预测测试 ===")
    test_inputs = [2.0, 3.0]
    prediction = lgp.predict(test_inputs)
    actual = target_function2(test_inputs[0], test_inputs[1])
    print(f"输入: {test_inputs}")
    print(f"预测值: {prediction:.4f}")
    print(f"实际值: {actual:.4f}")
    print(f"误差: {abs(prediction - actual):.4f}")

    return lgp

def plot_convergence(lgp_instances, titles):
    """绘制多个LGP实例的收敛曲线"""
    plt.figure(figsize=(12, 8))

    for i, (lgp, title) in enumerate(zip(lgp_instances, titles)):
        plt.plot(lgp.fitness_history, label=title, linewidth=2)

    plt.title('LGP算法收敛曲线对比', fontsize=16)
    plt.xlabel('代数', fontsize=14)
    plt.ylabel('最佳适应度', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数尺度更好地显示变化
    plt.tight_layout()
    plt.savefig('lgp_convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 运行简单示例
    lgp1 = example_simple()

    # 运行乘法示例
    lgp2 = example_multiplication()

    # 绘制收敛曲线对比
    plot_convergence([lgp1, lgp2], ['线性函数 f(x,y)=x+y', '乘法函数 f(x,y)=x*y'])