# 《遗传编程算法及其应用》

> 面向希望系统学习遗传编程、符号回归与进化计算的学生、教师和工程研究人员：从基础 GP/GEP 出发，逐步完成符号建模、分类、神经符号回归以及 GPU/并行实现。

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/SCUT-ELOG/GP-and-its-applications?display_name=tag&include_prereleases)](https://github.com/SCUT-ELOG/GP-and-its-applications/releases)
[![Issues](https://img.shields.io/github/issues/SCUT-ELOG/GP-and-its-applications)](https://github.com/SCUT-ELOG/GP-and-its-applications/issues)

**快速入口：** [10 分钟上手](#10-分钟运行第一个程序) · [章节代码](#章节算法代码数据集对应表) · [免费导读](#免费导读课) · [购买纸质书](#图书信息与购买) · [勘误](#勘误与更新) · [引用](#课程采用与学术引用) · [作者主页](https://jinghuizhong.com/) · [读者交流](https://github.com/SCUT-ELOG/GP-and-its-applications/issues)

**推荐引用：** 钟竞辉. 遗传编程算法及其应用[M]. 北京: 科学出版社, 2026.（[BibTeX 与代码引用](#课程采用与学术引用)）

> 本仓库提供可运行的配套代码、数据集和入门导读，不提供全书电子版。代码用于帮助读者验证算法、复现实验和继续研究，完整的理论推导、方法脉络与案例讲解请参阅纸质书。

## 为什么学习遗传编程

传统机器学习通常先由人选定模型结构，再从数据中估计参数；遗传编程（Genetic Programming, GP）则把程序结构或数学表达式本身也放入搜索空间，通过进化搜索自动发现可解释的规则。它尤其适合：

- 从数据中发现显式或隐式数学关系，即符号回归；
- 在准确率、复杂度、物理量纲等多个目标之间取得平衡；
- 自动构造分类规则、组合优化策略或可执行程序；
- 将进化搜索与最小二乘、神经网络及并行计算结合。

## 本书解决什么问题

本书围绕“如何表示程序、如何搜索程序、如何让结果更准确且可解释、如何扩展到真实任务”展开。读者可以从一套最小 GP 实现开始，逐步理解 GEP、LGP、GE、多目标优化与语义GP，并把方法用于符号回归、分类、有限元建模和神经符号回归；最后学习 CUDA、OpenMP 和 MPI 实现思路。

建议学习路径：

```text
基础表示与遗传操作（第 2 章）
        ↓
显式 / 隐式符号回归（第 3–4 章）
        ↓
多目标与组合优化（第 5–6 章）
        ↓
分类与神经符号回归（第 7–8 章）
        ↓
GPU 与并行实现（第 9 章）
```

## 10 分钟运行第一个程序

下面用第 2 章的标准遗传编程（SGP）拟合 `y = x²`。建议使用 Python 3.8–3.12，并为各章创建独立虚拟环境。

### 1. 获取代码并安装

```bash
git clone https://github.com/SCUT-ELOG/GP-and-its-applications.git
cd GP-and-its-applications
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e "GP-book-code/chapter 2/SGP"
```

Linux / macOS：

```bash
source .venv/bin/activate
python -m pip install -e "GP-book-code/chapter 2/SGP"
```

### 2. 运行最小示例

```bash
python "GP-book-code/chapter 2/SGP/quickstart.py"
```

**预期结果：** 终端输出进化得到的表达式和 R²，当前目录生成 `fitness_evolution.svg` 适应度曲线。遗传编程具有随机性，表达式形式与分数可能因环境而略有不同；固定 `seed=42` 有助于复现。更多拟合效果图和四组合成实验见 [`demo_basic.ipynb`](GP-book-code/chapter%202/SGP/notebooks/demo_basic.ipynb)。

安装 SGP 时会自动安装这个示例所需的 `deap`、`numpy`、`scikit-learn` 和 `matplotlib`，无需另外逐个安装。若你已经位于 `GP-book-code/chapter 2/SGP` 目录，也可以运行 `python quickstart.py`。

第 2 章根目录下的 [`grammatical_evolution.py`](GP-book-code/chapter%202/grammatical_evolution.py) 是独立的GE示例；运行它前请在仓库根目录执行 `python -m pip install deap`，再运行 `python "GP-book-code/chapter 2/grammatical_evolution.py"`。

## 章节—算法—代码—数据集对应表

| 章节 | 核心主题 / 算法 | 代码入口 | 配套数据或案例 | 运行后可观察到 |
| --- | --- | --- | --- | --- |
| 第 2 章 | GEP、LGP、SGP、GE | [chapter 2](GP-book-code/chapter%202/) | 合成函数、PMLB（Notebook 按需下载） | 表达式、预测误差、适应度曲线 |
| 第 3 章 | 显式符号回归 MLDEP | [`mldep.py`](GP-book-code/chapter%203/mldep.py) | [F1–F12 基准数据](GP-book-code/chapter%203/Benchmark/new_compact_sample_dataset/) | 显式表达式及训练/测试误差 |
| 第 4 章 | 隐式符号回归 LSE-GEP、CL-FEM | [chapter 4](GP-book-code/chapter%204/%E7%AC%AC%E5%9B%9B%E7%AB%A0/) | LSE-GEP 与 FEM 示例 | 隐式关系、最小二乘系数 |
| 第 5 章 | 多目标 GP | [chapter 5](GP-book-code/chapter%205/) | [Feynman 含量纲数据](GP-book-code/chapter%205/Feynman_with_units/) | 精度—复杂度权衡、量纲约束结果 |
| 第 6 章 | GP 组合优化 | [chapter 6](GP-book-code/chapter%206/) | 组合优化与表达式树案例 | 搜索过程及候选表达式树 |
| 第 7 章 | GP 分类 | [Python 版与快速指南](GP-book-code/chapter%207/python_version/) | MONK、Iris、Haberman、Zoo | 分类规则、混淆矩阵、准确率与召回率 |
| 第 8 章 | DL+GP PGGP、NeSymReS | [chapter 8 指南](GP-book-code/chapter%208/README.md) | 配置文件、Notebook 与模型案例 | 神经符号表达式和算法对比 |
| 第 9 章 | CUDA、OpenMP、MPI GP | [chapter 9](GP-book-code/chapter%209/) | CPU/GPU 与分布式实现 | 编译运行结果与并行性能 |

各章代码相对独立，语言、依赖和数据路径并不完全相同。请先阅读对应目录说明，不建议在仓库根目录一次性安装全部依赖。

## 可直接运行的案例

### 案例 A：标准 GP 符号回归

```bash
cd "GP-book-code/chapter 2/SGP"
python -m pip install -e ".[dev]"
jupyter notebook notebooks/demo_basic.ipynb
```

预期看到 `x²`、双变量加法、含噪多项式和三角函数四组实验，以及拟合曲线、残差和适应度变化图。详细接口见 [SGP 使用说明](GP-book-code/chapter%202/SGP/README.md)。

### 案例 B：线性遗传编程

```bash
cd "GP-book-code/chapter 2/LGP"
python -m pip install -r requirements.txt
python example_usage.py
```

预期看到线性函数与乘法函数的演化过程，以及测试输入、预测值、真实值和误差。详细说明见 [LGP README](GP-book-code/chapter%202/LGP/README.md)。

### 案例 C：GEP 分类

```bash
cd "GP-book-code/chapter 7/python_version"
python -m pip install -r requirements.txt
python main.py
```

按提示选择 MONK、Iris、Haberman 或 Zoo 数据集及运行模式。程序会输出数据规模、进化信息、分类规则、混淆矩阵、准确率和召回率。若只想快速检查环境，可运行 `python test_simple.py`。参见 [快速指南](GP-book-code/chapter%207/python_version/QUICK_START.md)。

### 第 8–9 章运行提示

- 第 8 章部分实验需要额外模型权重和数据集；请先阅读 [准备步骤与引用说明](GP-book-code/chapter%208/README.md)。
- 第 9 章 `.cu` 文件需要 CUDA 工具链；OpenMP/MPI 示例需要相应 C++ 编译器与运行环境，不能直接作为 Python 脚本运行。

## 免费导读课

这里提供三条不替代正文的配套导读路线，适合自学或作为课程预习材料：

| 导读 | 建议用时 | 学习任务 | 实践入口 |
| --- | ---: | --- | --- |
| 第 2 章：程序如何进化 | 60–90 分钟 | 对比树、线性和基因表达式三种表示；识别选择、交叉、变异的作用 | [SGP 基础 Notebook](GP-book-code/chapter%202/SGP/notebooks/demo_basic.ipynb) / [LGP 示例](GP-book-code/chapter%202/LGP/example_usage.py) |
| 第 3 章：从数据到公式 | 60–90 分钟 | 区分参数优化与结构搜索；在 F1–F12 上观察训练/测试误差 | [MLDEP 代码](GP-book-code/chapter%203/) / [基准数据](GP-book-code/chapter%203/Benchmark/new_compact_sample_dataset/) |
| 第 7 章：从回归到分类 | 60–90 分钟 | 理解表达式如何转化为分类规则；比较 GEP 与传统分类器 | [GEP 分类快速指南](GP-book-code/chapter%207/python_version/QUICK_START.md) |

建议每次导读完成三个动作：先读本章的“问题与表示”，再运行默认案例，最后只修改一个参数并记录结果。后续课件、讲解视频或讲义会在 [Releases](https://github.com/SCUT-ELOG/GP-and-its-applications/releases) 中按版本发布。

## 图书信息与购买

| 项目 | 信息 |
| --- | --- |
| 书名 | 《遗传编程算法及其应用》 |
| 作者 | 钟竞辉 |
| 出版社 | [科学出版社](https://www.sciencep.com/) |
| 出版年份 | 2026 |
| 图书封面 / 目录 | 待出版社最终资料确认后补充 |
| 京东购买 | [在京东搜索本书](https://search.jd.com/Search?keyword=%E9%81%97%E4%BC%A0%E7%BC%96%E7%A8%8B%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E5%BA%94%E7%94%A8)（请以正式上架信息为准） |

本仓库未收录全书 PDF。若你希望系统理解算法推导、设计选择和完整案例，请购买正版纸质书；仓库代码用于随书实践与版本更新。

## 勘误与更新

- 发现书中或代码中的问题，请提交 [Issue](https://github.com/SCUT-ELOG/GP-and-its-applications/issues/new)，注明章节、页码或文件路径、原内容及建议修改。
- 使用中遇到问题，请先搜索 [已有 Issues](https://github.com/SCUT-ELOG/GP-and-its-applications/issues)，再提供操作系统、Python/编译器版本、完整命令与错误信息。
- 稳定快照、更新说明和新增教学资料将在 [Releases](https://github.com/SCUT-ELOG/GP-and-its-applications/releases) 发布。当前尚无正式稳定版时，请使用默认分支并记录具体 commit。

## 课程教学资源

欢迎高校教师将本书用于进化计算、智能优化、机器学习或符号回归课程。仓库现有 Notebook、章节案例和数据集可用于课堂演示与实验作业。


## 课程采用与学术引用

### 引用图书

```text
钟竞辉. 遗传编程算法及其应用[M]. 北京: 科学出版社, 2026.
```

```bibtex
@book{zhong2026geneticprogramming,
  author    = {钟竞辉},
  title     = {遗传编程算法及其应用},
  publisher = {科学出版社},
  address   = {北京},
  year      = {2026}
}
```

### 引用代码

在软件 DOI 发布前，可引用仓库与所使用的 commit 或 Release：

```text
Jinghui Zhong. GP and Its Applications: Companion Code.
https://github.com/SCUT-ELOG/GP-and-its-applications, accessed YYYY-MM-DD, commit <SHA>.
```

本仓库计划通过 Zenodo 归档 GitHub Release 并申请 DOI；DOI 生成后将在顶部徽章和本节更新。参见 [GitHub 官方的 Zenodo 引用指南](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)。

## 许可证与贡献

代码采用 [MIT License](LICENSE)。欢迎通过 Issue 报告问题；提交修改前，请将改动限定在明确的章节，并说明运行环境、复现命令与验证结果。

---

如果这套代码帮助你完成了课程、实验或研究，欢迎 Star 仓库，并在论文、报告或课程材料中引用本书与对应代码版本。
