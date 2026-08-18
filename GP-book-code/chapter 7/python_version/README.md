# GEP Classification - Python Version

这是基因表达式编程（Gene Expression Programming, GEP）分类器的Python实现版本，从C++项目转换而来。

## 项目简介

本项目实现了GEP算法用于分类任务，并与传统机器学习算法（决策树、朴素贝叶斯、随机森林、SVM）进行性能比较。

## 功能特点

- **GEP算法**: 完整实现基因表达式编程算法
- **多种数据集支持**: MONK、Iris、Haberman等数据集
- **算法对比**: 与sklearn的经典算法进行性能比较
- **面向对象设计**: 清晰的模块化架构

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```python
from datasets.monk_dataset import Monk1
from main import run_experiments

# 选择数据集
dataset = Monk1()

# 运行实验
run_experiments(dataset)
```

## 项目结构

```
python_version/
├── gep.py              # GEP核心算法实现
├── dataset_abstract.py # 数据集抽象基类
├── datasets/           # 具体数据集实现
│   ├── monk_dataset.py
│   ├── iris_dataset.py
│   └── haberman_dataset.py
├── main.py             # 主程序入口
└── requirements.txt    # 依赖包
```

## GEP算法参数

- 种群规模: 1000
- 最大迭代次数: 2000
- 染色体长度: 100
- 交叉率: 0.50
- 变异率: 0.10
- 旋转率: 0.02

## 函数集

支持12种操作符：
- 条件运算: IF (三目运算符)
- 算术运算: +, -, *, /
- 逻辑运算: OR, AND
- 数学函数: SQRT, EXP, LOG, SIN, COS

## 作者

从C++原项目转换而来
