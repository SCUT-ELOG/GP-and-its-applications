# Transformer辅助遗传编程用于符号回归

本项目实现了Transformer辅助遗传编程（PGGP）方法，用于符号回归任务。

## 模型权重和数据集下载

### 模型权重
1. 创建 `weights` 文件夹
2. 从以下链接下载Transformer模型的预训练权重：
   `https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/tree/main`
3. 将下载的权重文件放入 `weights` 文件夹

### 数据集
1. 创建 `benchmark_dataset` 文件夹
2. 从以下链接下载Feynman数据集：
   `https://space.mit.edu/home/tegmark/aifeynman.html`
3. 将下载的数据集文件放入 `dataset` 文件夹

## 算法项目介绍

### 1. PGGP (Transformer辅助遗传编程)
- **文件**: `pggp.py`
- **简介**: 预训练模型引导的遗传编程方法，利用Transformer模型辅助GP的初始化和变异过程，提高符号回归效率和准确性。

### 2. stGP (标准遗传编程)
- **文件**: `stGP.py`
- **简介**: 标准遗传编程实现，不使用Transformer辅助，作为基线方法与PGGP进行比较。

### 3. NeSymReS (神经网络符号回归)
- **文件**: `NeSymReS.py`
- **简介**: 基于神经网络的符号回归方法，使用深度学习模型直接学习符号表达式。

### 4. 算法比较工具
- **文件**: `compare_algorithms.py`
- **简介**: 用于比较不同算法性能的可视化工具。该脚本可以加载PGGP和stGP算法的运行结果，计算统计指标（中位数、四分位数），并生成对比图表。支持输出为SVG和PDF格式的矢量图，便于在论文和报告中使用。

## 环境配置

本项目使用uv作为包管理器。为了方便用户快速配置环境，我们提供了requirements.txt文件，包含了所有必要的依赖包。

### 安装步骤

1. 安装uv包管理器（如果尚未安装）：
   ```bash
   pip install uv
   ```

2. 使用uv创建虚拟环境并安装依赖：
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/Mac
   # 或者在Windows上使用 .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. 或者，如果您想直接在当前环境中安装所有依赖：
   ```bash
   uv pip install -r requirements.txt
   ```

## 快速使用

运行pggp实验（另外两个类似）：
```bash
python pggp.py
```

如需了解更多详细信息，请参考原项目文档。

## 引用

```
@article{han2025transformer,
  title={Transformer-Assisted Genetic Programming for Symbolic Regression [Research Frontier]},
  author={Han, Xiaoxu and Zhong, Jinghui and Ma, Zhitong and Mu, Xin and Gligorovski, Nikola},
  journal={IEEE Computational Intelligence Magazine},
  volume={20},
  number={2},
  pages={58--79},
  year={2025},
  publisher={IEEE}
}
```
