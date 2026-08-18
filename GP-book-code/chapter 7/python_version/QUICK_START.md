# 快速开始指南

## 安装依赖

在运行项目之前，请先安装所需的Python包：

```bash
cd python_version
pip install -r requirements.txt
```

## 项目结构

```
python_version/
├── gep.py                      # GEP核心算法（染色体、遗传操作、适应度计算）
├── dataset_abstract.py         # 数据集抽象基类
├── datasets/                   # 数据集实现
│   ├── __init__.py
│   ├── monk_dataset.py        # MONK-1/2/3数据集
│   └── other_datasets.py      # Iris、Haberman、Zoo数据集
├── main.py                     # 主程序（交互式）
├── test_simple.py             # 简单测试脚本
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明
```

## 运行方式

### 方式1：交互式运行（推荐）

```bash
cd python_version
python main.py
```

然后按照提示：
1. 选择数据集（1-6）
2. 选择运行模式（传统算法/GEP/全部）

### 方式2：简单测试（快速验证）

```bash
cd python_version
python test_simple.py
```

这将直接在MONK-1数据集上运行GEP算法。

### 方式3：编程方式

```python
from datasets.monk_dataset import Monk1
from sklearn.tree import DecisionTreeClassifier

# 创建数据集
dataset = Monk1()

# 运行决策树
dtree = DecisionTreeClassifier(max_depth=10)
dataset.train_and_test(dtree, "Decision Tree")

# 运行GEP
dataset.train_gep_and_test()
```

## 支持的数据集

| 数据集 | 类别数 | 训练样本 | 测试样本 | 特征数 | 类 |
|--------|--------|----------|----------|--------|-----|
| MONK-1 | 2 | 124 | 432 | 15 | Monk1 |
| MONK-2 | 2 | 169 | 432 | 15 | Monk2 |
| MONK-3 | 2 | 122 | 432 | 15 | Monk3 |
| Iris | 3 | 118 | 30 | 4 | Iris |
| Haberman | 2 | 227 | 62 | 3 | HabermanSurvival |
| Zoo | 7 | 75 | 26 | 16 | Zoo |

## GEP算法参数

- **种群规模**: 1000
- **最大迭代**: 2000代
- **染色体长度**: 100
- **交叉率**: 0.50
- **变异率**: 0.10
- **旋转率**: 0.02

## 函数集

支持12种操作符：
- **条件**: IF (三目运算符)
- **算术**: +, -, *, /
- **逻辑**: OR, AND
- **数学**: SQRT, EXP, LOG, SIN, COS

## 预期输出

程序将输出：
1. 数据加载信息
2. 每代的进化信息（适应度、覆盖情况）
3. 规则生成信息（MDL值、规则数）
4. 后剪枝信息（规则顺序、默认类别）
5. 测试结果（混淆矩阵、准确率、召回率）

## 注意事项

1. **数据文件路径**: 数据文件应位于项目根目录（`GP_classification`），与`python_version`文件夹同级
2. **运行时间**: GEP算法运行时间较长（特别是初次运行），请耐心等待
3. **内存使用**: 大种群规模可能消耗较多内存
4. **随机性**: 由于算法的随机性，每次运行结果可能略有不同

## 与C++版本的差异

1. **语言特性**: 使用Python的动态类型和NumPy向量化操作
2. **性能**: Python版本比C++版本慢，但代码更简洁易读
3. **依赖库**: 
   - C++版本使用OpenCV的机器学习模块
   - Python版本使用scikit-learn
4. **接口**: Python版本提供了更友好的交互式界面

## 调试技巧

如果遇到问题：

1. **检查数据文件**: 确保所有数据文件在正确位置
2. **查看详细输出**: 程序会打印详细的运行信息
3. **减小参数**: 可以修改`gep.py`中的参数来加快测试
   ```python
   POPULATION_SIZE = 100  # 减小种群
   GENERATION_LIMIT = 200  # 减少迭代
   ```

## 扩展新数据集

要添加新数据集，继承`DatasetAbstract`类：

```python
from dataset_abstract import DatasetAbstract
import numpy as np

class MyDataset(DatasetAbstract):
    def __init__(self):
        super().__init__("train.data", classes=2, samples=100, features=10)
        self.INPUT_TEST_NAME = "test.data"
        self.INPUT_TEST_NUM = 50
        self.load_train_data()
    
    def load_train_data(self):
        # 实现数据加载逻辑
        self.input_attrs = ...  # numpy array [samples, features]
        self.input_class = ...  # numpy array [samples]
    
    def load_test_data(self):
        # 实现测试数据加载逻辑
        return test_attrs, test_class
```

## 性能优化建议

如果需要提高性能：

1. 使用NumPy的向量化操作（已实现）
2. 减小种群规模和迭代次数
3. 使用JIT编译器（如Numba）
4. 考虑使用多进程并行化

## 贡献

欢迎提交问题和改进建议！
