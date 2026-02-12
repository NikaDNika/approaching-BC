# Approaching Behaviour Clone

## 1. Introduction

本项目实现了一个基于 Transformer 的 7 自由度机械臂（Franka Panda + Mobile Base）行为克隆（BC）系统。
项目包含大规模专家数据生成、Transformer 策略预训练



## 2. Requirements

### 操作系统及软硬件版本配置要求

**OS:** Ubuntu 22.04

**GPU:** NVIDIA RTX 4090

**CUDA:** 11.8/12.1

### 环境安装

#### 方式一：Conda（推荐）

首先查看项目中的enviroment.yml，替换“<your_conda_env_name>为您想创建的虚拟环境名字，保存。
然后cd到项目根目录，在终端执行

```
conda env create -f enviroment.yml
conda activate <your_conda_env_name>
```

#### 方式二：Pip

或者用您自己的虚拟环境，执行

```
pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
```



## 3. 快速开始

### 查看仿真场景

注意在configs/xx.xml中修改路径

```
simulate configs/fixed_scene_living_room.xml
simulate configs/fixed_scene_kitchen.xml
simulate configs/fixed_scene_storage.xml
simulate configs/fixed_scene_corner.xml
```

### 查看目标生成区域

待抓取物体与放置位置的生成位置用同一套集合

```
python -m tools.viz_target_distribution
```

关闭仿真窗口会依次展示四个场景的目标生成区域，默认200个采样点。

如需修改目标生成区域，可在 **src/env/scene_generator.py** 中修改对应场景的sampling_regions.append()方法中的参数。

### 数据采集

使用cuRobo规划期生成专家演示数据

```
python -m src.data.collector
```

可视化最新采集的数据

```
python -m tools.replay
```

### 策略预训练

用行为克隆训练基础的Transformer策略，学习“抓取与放置”任务

```
python -m src.training.train_bc_with_ar_transfoermer_multistep
```

训练完成后将在 checkpoints/ 下生成 bc_with_ar_transformer_multistep_xxxx_xxx 文件夹，其中的 .pth 即为训练好的模型文件。

可视化训练好的模型

```
python -m tools.rollout_closedloop_bc_with_ar_from_npz \
  --data-dir /media/j16/8deb00a4-cceb-e842-a899-55532424da473/dataset_v2 \
  --scenario living_room \
  --policy-ckpt checkpoints/bc_with_ar_transformer_multistep_xxx/best_model.pth \
  --num-points 2048 \
  --max-steps 300 \
  --device cuda \
  --control-dt 0.05
```

### 在线微调

加载预训练权重，开启“上帝视角”教师进行避障特训。

注意修改 src/training/main_finetune.py 中的 CKPT_PATH 为你自己的模型

```
python -m src.training.main_finetune
```

可视化仍可用 tools/rollout_closedloop_bc_with_ar_from_npz.py

