# SGP — Standard Genetic Programming for Symbolic Regression

基于 DEAP 框架的标准遗传编程符号回归库，提供 sklearn 兼容的 `fit` / `predict` / `score` 接口，可作为 baseline 直接调用。

## 项目结构

```
SGP/
├── sgp/
│   ├── __init__.py       # 公开 API（导出 SGP）
│   ├── _gp.py            # 核心 GP 引擎（DEAP 驱动）
│   └── _estimator.py     # sklearn 风格封装
├── notebooks/
│   ├── demo_basic.ipynb  # 基础用法（4 个合成实验）
│   └── demo_pmlb.ipynb   # PMLB 基准测试（8 个真实数据集）
└── pyproject.toml
```

## 安装

```bash
git clone <repo-url> && cd SGP
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e . -i https://mirrors.aliyun.com/pypi/simple/

# PMLB 基准测试需要额外依赖
uv pip install pmlb -i https://mirrors.aliyun.com/pypi/simple/
```

## 快速上手

```python
from sgp import SGP
import numpy as np

X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = (X ** 2).flatten()

model = SGP(seed=42).fit(X, y)
print(model.best_expression_)  # 发现的表达式
print(model.score(X, y))       # R²
print(model.predict(X))        # 预测值

# 保存 fitness 曲线（矢量图 SVG）
model.plot_fitness()                     # 默认 figures/fitness_evolution.svg
model.plot_fitness("figures/my.svg")     # 自定义路径
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pop_size` | 500 | 种群大小 |
| `generations` | 40 | 最大进化代数 |
| `cxpb` | 0.7 | 交叉概率 |
| `mutpb` | 0.1 | 变异概率 |
| `elite_size` | 5 | 精英保留数量 |
| `tournament_size` | 7 | 锦标赛选择大小 |
| `max_depth` | 17 | 树最大深度（Koza 推荐） |
| `init_depth` | (1, 2) | 初始化树深度范围 |
| `complexity_weight` | 0.01 | 复杂度惩罚权重 |
| `function_set` | None | 自定义函数集（None=全部） |
| `patience` | 20 | 早停耐心值（0=不启用） |
| `seed` | None | 随机种子 |
| `verbose` | True | 打印进化进度 |

## 函数集

默认: `add`, `sub`, `mul`, `protected_div`, `sin`, `cos`, `protected_log`, `protected_exp`, `protected_sqrt`

保护性运算对除零、负数开方、溢出等做了安全处理（`np.where` 实现，无 try-except）。

```python
# 自定义函数子集
model = SGP(function_set=["add", "mul", "sin", "cos"])
```

## 训练后属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `best_expression_` | str | 最佳表达式字符串 |
| `best_func_` | callable | 预测函数 `f(X) → y` |
| `best_fitness_` | float | 最佳适应度值 |
| `best_complexity_` | int | 表达式节点数 |
| `history_` | dict | fitness 历史 `{min, avg, max}` |
| `n_generations_` | int | 实际进化代数 |
| `n_features_in_` | int | 输入特征数 |

## Notebook Demo

```bash
cd notebooks

# 基础用法：x²、x₀+x₁、噪声多项式、三角函数
jupyter notebook demo_basic.ipynb

# PMLB 基准：8 个真实回归数据集（vineyard、cpu、pm10 等）
# 首次运行自动下载数据集到 pmlb_data/，后续秒加载
jupyter notebook demo_pmlb.ipynb
```

### demo_basic — 4 个合成实验

| 实验 | 目标函数 | 可视化 |
|------|---------|--------|
| 1 | $y = x^2$ | 拟合对比 + fitness 曲线 |
| 2 | $y = x_0 + x_1$ | y_true vs y_pred 散点图 |
| 3 | $y = x^3 - 2x^2 + x$ + 噪声 | 拟合对比 + 残差分布 |
| 4 | $y = \sin(x) + \cos(2x)$ | 函数拟合 + 误差带 |

### demo_pmlb — 8 个 PMLB 真实数据集

| 数据集 | PMLB 名称 | 样本数 | 特征数 |
|--------|-----------|--------|--------|
| vineyard | 192_vineyard | 52 | 2 |
| elusage | 228_elusage | 55 | 2 |
| salary | 1096_FacultySalaries | 50 | 4 |
| ESL | 1027_ESL | 488 | 4 |
| cloud | 210_cloud | 108 | 5 |
| vinnie | 519_vinnie | 380 | 2 |
| cpu | 230_machine_cpu | 209 | 6 |
| pm10 | 522_pm10 | 500 | 7 |

包含 R² 柱状图、复杂度 vs 准确率散点图、运行时间分布、表达式对比表。

## 核心实现

- **进化策略**: 精英保留 + 锦标赛选择 + 交叉(`cxOnePoint`) + 变异(`mutUniform`)
- **深度限制**: `staticLimit(max=17)` 防止 bloat
- **编译缓存**: `str(individual)` 作为 key，避免重复 `gp.compile`
- **早停**: `patience > 0` 时连续 N 代无改善则停止
- **适应度**: RMSE + `complexity_weight × 节点数`

## 依赖

- deap >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- pmlb >= 1.0.0（可选，PMLB 基准测试用）

## 许可证

MIT
