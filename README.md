# 遗传编程算法及其应用：配套代码

本仓库是《遗传编程算法及其应用》的配套源代码，覆盖从基础遗传编程、基因表达式编程，到符号回归、符号分类、神经符号回归以及 GPU/并行实现的多个示例。

仓库中的各章节代码相对独立，使用的语言和依赖也不完全相同。请进入对应章节目录后，按照该章节的说明安装依赖和运行示例。

## 内容概览

| 章节 | 主题 | 主要内容 |
| --- | --- | --- |
| 第 2 章 | 基础遗传编程 | GEP、LGP、标准遗传编程（SGP）和语法进化 |
| 第 3 章 | MLDEP | 基于基因表达式和依赖分析的符号建模、矩阵化简与数据集 |
| 第 4 章 | LSE-GEP | 最小二乘估计与 GEP 的结合，包括 FEM 示例 |
| 第 5 章 | 多目标与语义遗传编程 | 多目标 GP、语义操作、维度间隙解析树与 Feynman 数据 |
| 第 6 章 | GP 变体 | CCGP、HBWS 和随机表达式树生成 |
| 第 7 章 | GEP 分类 | GEP 分类器及 MONK、Iris、Haberman、Zoo 数据集 |
| 第 8 章 | 神经符号回归 | PGGP、标准 GP、NeSymReS 和算法对比实验 |
| 第 9 章 | 高性能实现 | CUDA、OpenMP 和 MPI 相关的 GP 实现 |

## 目录结构

```text
GP-book-code/
├── chapter 2/
│   ├── GEP/                 # 基因表达式编程
│   ├── LGP/                 # 线性遗传编程
│   ├── SGP/                 # 标准遗传编程 Python 包
│   └── grammatical_evolution.py
├── chapter 3/               # MLDEP 及符号建模工具
├── chapter 4/第四章/         # LSE-GEP、CL-FEM
├── chapter 5/               # 多目标、语义 GP 与 Feynman 数据
├── chapter 6/               # GP 变体和表达式树生成
├── chapter 7/               # GEP 分类及分类数据集
├── chapter 8/               # PGGP、stGP、NeSymReS
└── chapter 9/               # CUDA、OpenMP、MPI 源代码
```

## 快速开始

### 第 2 章：SGP

SGP 是一个带有 sklearn 风格 `fit`、`predict` 和 `score` 接口的符号回归实现。

```powershell
cd "GP-book-code/chapter 2/SGP"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -c "from sgp import SGP; print(SGP)"
```

Linux/macOS 可将虚拟环境激活命令替换为：

```bash
source .venv/bin/activate
```

基础示例和 PMLB 基准实验位于 `chapter 2/SGP/notebooks/`。运行 PMLB 示例前，还需要安装可选依赖：

```bash
python -m pip install -e ".[dev]"
```

### 第 2 章：LGP

```bash
cd "GP-book-code/chapter 2/LGP"
python -m pip install -r requirements.txt
python example_usage.py
```

### 第 7 章：GEP 分类

```bash
cd "GP-book-code/chapter 7/python_version"
python -m pip install -r requirements.txt
python main.py
```

该目录包含 `gep.py`、数据集抽象类、多个数据集实现和实验入口 `main.py`。原始数据文件位于 `chapter 7/` 下。

### 第 8 章：神经符号回归

```bash
cd "GP-book-code/chapter 8"
python -m pip install -r requirements.txt
python pggp.py
```

第 8 章的实验可能需要额外的模型权重和数据集。详细准备步骤、配置文件和引用信息请参阅 [第 8 章 README](GP-book-code/chapter%208/README.md)。

## 其他章节

- 第 2 章 GEP：入口文件为 `GP-book-code/chapter 2/GEP/main.py`。
- 第 2 章语法进化：入口文件为 `GP-book-code/chapter 2/grammatical_evolution.py`。
- 第 3 章：主要入口和参数定义见 `mldep.py`、`constants.py`，运行前请根据代码中的数据路径配置数据集。
- 第 4 章：Python 脚本位于 `GP-book-code/chapter 4/第四章/`，包括 `LSE-GEP.py`、`functional-programming-LSE-GEP.py` 和 `CL-FEM.py`。
- 第 5 章：运行 `MOGP.py`、`semantic.py` 等脚本前，请检查 Feynman 数据文件的路径和所需 Python 包。
- 第 6 章：`CCGP.py`、`HBWS.py` 和 `RandomExpressionTreeGeneration.py` 是相互独立的实验脚本。
- 第 9 章：`.cu` 文件需要 CUDA 工具链；OpenMP/MPI 示例需要对应的 C++ 编译器和运行环境，不能通过 Python 直接运行。

## 环境建议

- Python 项目建议使用 Python 3.8 或更高版本，并为每个章节创建独立虚拟环境。
- 依赖文件只适用于相应章节，不建议在仓库根目录一次性安装所有依赖。
- Jupyter Notebook 示例需要安装 `jupyter` 或 `jupyterlab`。
- 部分实验涉及 CUDA、PyTorch、MPI 或 OpenMP，请先确认本机已安装匹配版本的工具链。

## 数据与大文件

仓库包含符号回归基准数据、分类数据和 Feynman 数据。第 5 章部分数据文件体积较大，GitHub 对其给出大文件提示；如果后续继续维护这些文件，建议使用 [Git LFS](https://git-lfs.github.com/) 管理。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。

## 引用

如果本仓库或第 8 章的 Transformer 辅助遗传编程代码用于学术研究，请同时参考对应章节文档中的论文和引用信息。