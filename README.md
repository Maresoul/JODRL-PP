## 项目名称

Repository for 'Privacy-Preserving Offloading Scheme in Multi-Access Edge Computing Based on MADRL'

## 实验环境

- Python版本: 3.8.5
- PyTorch版本: 1.8.1
- GPU运行：是

## 简介

本算法是对论文Privacy-Preserving Offloading Scheme in Multi-Access Edge Computing Based on MADRL所提出模型和算法的代码实现，智能体通过与环境不断交互，获得奖励，逐渐学习更好的卸载策略，从而完成对模型的性能和隐私优化。

## 运行方法

1. 安装依赖：确保您已经安装了Python 3.8.5和PyTorch 1.8.1，并准备了支持GPU运行的环境。

2. 克隆项目：使用以下命令将项目克隆到本地：
git clone https://github.com/Maresoul/JODRL-PP

3. 切换目录：进入项目文件夹:
cd pytorch-jodrl_pp

4. 运行程序：执行以下命令运行主程序
python main.py

5.对照算法：
进入compare_algorithm文件夹，找到相应算法执行

## 文件结构

- `main.py`: 主程序入口，运行实验设计程序。
- `compare_algorithm/`: 包含对照试验的程序文件夹。
- `discen.py`: 基线算法1，去中心化学习策略网络。
- `local.py`: 基线算法2，任务全部本地执行。
- `near.py`: 基线算法，任务卸载到最近边缘节点。
- `qmix`: QMIX算法程序文件夹。
- `mappo/`: MAPPO算法程序文件夹。
- `MEC_env.py`: 与智能体交互的MEC环境设置。

## 使用说明

程序经过多次修改和更正，在本机能够正常运行，如果运行出错，很可能GPU参数有误，请仔细确认。








