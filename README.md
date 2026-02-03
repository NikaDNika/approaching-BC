# Deep Reactive Policy (DRP) for Mobile Manipulation

An unofficial implementation and extension of the Deep Reactive Policy (DRP) framework for mobile manipulator planning in dynamic environments. This project integrates MuJoCo physics simulation, cuRobo motion planning, and transformer-based policy learning (IMPACT).

![Project Structure](https://img.shields.io/badge/Project-DRP_Full-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 📂 Project Structure

```text
DRP_FULL/
├── checkpoints/            # Trained model weights & configs
├── configs/                # XML configurations for robot & scenes
│   ├── mobile_panda.xml    # Robot definition (Mobile Base + Franka Emika Panda)
│   └── ...                 # Scene definitions
├── dataset_v2/             # Generated dataset (RGB-D + PointCloud + Trajectory)
├── mujoco_menagerie/       # Third-party robot assets
├── src/                    # Source code
│   ├── data/               # Data collection & loading
│   │   ├── collector.py    # Main script for data generation
│   │   └── dataset.py      # PyTorch Dataset
│   ├── env/                # MuJoCo environment wrappers
│   │   ├── mujoco_env.py   # Simulation & Rendering
│   │   └── scene_generator.py # Procedural scene generation
│   ├── evaluation/         # Evaluation scripts
│   │   └── evaluate_drp.py # Closed-loop policy rollout
│   ├── models/             # Neural Networks
│   │   ├── impact.py       # IMPACT Policy (Transformer)
│   │   └── modules.py      # PointNet++ Encoder
│   ├── planning/           # Motion Planning (Oracle)
│   │   └── planner.py      # NVIDIA cuRobo wrapper
│   ├── training/           # Training scripts
│   │   └── train_bc.py     # Behavior Cloning training
│   └── utils/              # Utilities
└── tools/                  # Helper scripts (visualization, debugging)
    ├── replay.py           # Replay generated data
    ├── check_data.py       # Inspect npz files
    └── tune_camera.py      # Camera pose adjustment tool
```

---

## 🛠️ Installation

### 1. Prerequisites
*   Ubuntu 20.04+
*   Python 3.8+
*   NVIDIA GPU (RTX 3070+ recommended) with CUDA 11.8+
*   [MuJoCo 2.3.7+](https://github.com/google-deepmind/mujoco)

### 2. Setup Environment
```bash
# Create conda environment
conda create -n drp python=3.10
conda activate drp

# Install PyTorch (adjust cuda version accordingly)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install mujoco opencv-python scipy pyyaml tqdm matplotlib

# Install NVIDIA cuRobo (Follow official instructions)
# https://curobo.org/get_started/1_install_instructions.html
pip install curobo
```

### 3. Install Source Package
Run this from the project root to enable absolute imports:
```bash
pip install -e .
```

---

## 🚀 Usage Guide

### 1. Data Collection (Generation)
Generate expert trajectories using cuRobo planner in procedurally generated scenes (Living Room, Kitchen, Storage, Corner).

```bash
# Run data collector (Multi-process by default)
# Ensure you have set MUJOCO_GL=egl if running on headless server
python -m src.data.collector
```
*   **Output**: Data will be saved to `dataset_v2/`.
*   **Features**: Dynamic base placement, multi-pose grasping attempt, RGB-D rendering, and point cloud fusion.

### 2. Data Inspection
Verify the generated data before training.

```bash
# Replay 3D trajectory in MuJoCo viewer
python tools/replay.py --scene living_room

# Inspect RGB-D images and Point Clouds
python tools/check_data.py
```

### 3. Training (Behavior Cloning)
Train the IMPACT policy (PointNet++ Encoder + Transformer Decoder).

```bash
# Run training script
python -m src.training.train_bc
```
*   **Checkpoints**: Saved to `checkpoints/drp_bc_baseline_{timestamp}/`.
*   **Config**: Hyperparameters are saved in `config.yaml`.

### 4. Evaluation
Evaluate the trained policy in a closed-loop simulation.

```bash
# Evaluate the best model from the latest experiment
python -m src.evaluation.evaluate_drp
```
*   The robot will attempt to reach the target (green sphere) using only point cloud observations.
*   **Obstacle Avoidance**: The policy should reactively avoid obstacles.

---

## 🔧 Configuration & Tuning

### Robot & Scene
*   **Robot XML**: `configs/mobile_panda.xml`. Modify this to change camera positions or robot physics.
*   **Scene Generation**: `src/env/scene_generator.py`. Adjust furniture layout and target sampling regions here.

### Cameras
To adjust camera viewpoints (Top, Side, Wrist):
1.  Edit `<camera>` tags in `configs/mobile_panda.xml`.
2.  Run visualization tool to verify:
    ```bash
    python tools/tune_camera.py
    ```
    Press `r` in the window to reload XML changes instantly.

### Planning (Oracle)
*   **Planner Config**: `src/planning/planner.py`.
*   **Grasp Offset**: `src/data/collector.py`. Adjust `real_target_world` calculation if the robot stops too far/close to the object.

---

## 📝 Notes
*   **Rendering Backend**: On Linux, ensure `export MUJOCO_GL=egl` is set to avoid OpenGL context conflicts during multi-process data collection.
*   **Coordinate Systems**:
    *   **World Frame**: Global MuJoCo world.
    *   **Base Frame (Link0)**: Robot base. Point clouds and actions are normalized to this frame.
*   **Gripper**: Currently disabled (visual only) to ensure simulation stability.

## 🤝 Acknowledgements
*   [Deep Reactive Policy (DRP)](https://arxiv.org/abs/xxxx)
*   [NVIDIA cuRobo](https://curobo.org/)
*   [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
```