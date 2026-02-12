# Approaching Behavior Cloning

## 1. Introduction

This project implements a Transformer-based Behavior Cloning (BC) system for a 7-DOF robotic arm (Franka Panda + Mobile Base). The project includes modules for large-scale expert data generation, Transformer policy pre-training, and policy fine-tuning.



## 2. Requirements

### OS and Hardware Requirements

**OS:** Ubuntu 22.04

**GPU:** NVIDIA RTX 4090

**CUDA:** 11.8 / 12.1

### Installation

#### Option 1: Conda (Recommended)

First, check the `environment.yaml` file in the project directory. Replace `<your_conda_env_name>` with the name you wish to use for your virtual environment, then save the file.
Next, navigate to the project root directory and run the following commands in your terminal:

```bash
conda env create -f environment.yaml
conda activate <your_conda_env_name>
```

#### Option 2: Pip

Alternatively, you can use your own virtual environment and run:

```bash
pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
```



## 3. Quick Start

### View Simulation Scenes

**Note:** Please check and modify paths in `configs/xx.xml` if necessary.

```bash
simulate configs/fixed_scene_living_room.xml
simulate configs/fixed_scene_kitchen.xml
simulate configs/fixed_scene_storage.xml
simulate configs/fixed_scene_corner.xml
```

If you need to generate simulation scenes yourself, the raw IKEA furniture `.glb` files are located in `assets/ikea_raw`, and the processed `.obj` files are in `assets/ikea_processed`.

### View Target Generation Areas

The generation zones for graspable objects and placement locations share the same set of definitions.

```bash
python -m tools.viz_target_distribution
```

Closing the simulation window will sequentially display the target generation areas for the four scenarios (default: 200 sampling points).

To modify the target generation areas, adjust the parameters in the `sampling_regions.append()` method within **src/env/scene_generator.py** for the corresponding scene.

### Data Collection

Use the cuRobo planner to generate expert demonstration data:

```bash
python -m src.data.collector
```

Visualize the latest collected data:

```bash
python -m tools.replay
```

### Policy Pre-training

Train a base Transformer policy using Behavior Cloning to learn "pick and place" tasks:

```bash
python -m src.training.train_bc_with_ar_transfoermer_multistep
```

Once training is complete, a folder named `bc_with_ar_transformer_multistep_xxxx_xxx` will be generated in `checkpoints/`. The `.pth` file inside is your trained model.

Visualize the trained model:

```bash
python -m tools.rollout_closedloop_bc_with_ar_from_npz \
  --data-dir your_dataset_path \
  --scenario living_room \
  --policy-ckpt checkpoints/bc_with_ar_transformer_multistep_xxx/best_model.pth \
  --num-points 2048 \
  --max-steps 300 \
  --device cuda \
  --control-dt 0.05
```

### Online Fine-tuning

Load the pre-trained weights and enable a "God-view" teacher for obstacle avoidance training.

**Note:** Modify `CKPT_PATH` in `src/training/main_finetune.py` to point to your specific model.

```bash
python -m src.training.main_finetune
```

For visualization, you can still use `tools/rollout_closedloop_bc_with_ar_from_npz.py`.
```