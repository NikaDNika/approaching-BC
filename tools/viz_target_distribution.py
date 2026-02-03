import numpy as np
import mujoco
import mujoco.viewer
import os
import time
from src.env.scene_generator import SceneGenerator

def visualize_distribution(scenario_type, n_samples=100):
    print(f"Generating distribution map for: {scenario_type}...")
    
    generator = SceneGenerator()
    dummy_xml = "temp_dist_gen.xml"
    
    # 1. 收集 N 个样本点
    # 注意：generate_target 每次也会微调家具位置(Jitter)。
    # 为了可视化，我们将所有目标点画在“第0次”生成的家具布局上。
    # 虽然家具位置有微小偏差(+/- 5cm)，但这足以看清目标点的分布区域。
    
    targets = []
    print(f"Sampling {n_samples} targets...", end="", flush=True)
    for i in range(n_samples):
        # 使用不同的随机种子生成不同的目标
        pos = generator.generate_target(scenario_type, dummy_xml, seed=i)
        targets.append(pos)
        if i % 10 == 0: print(".", end="", flush=True)
    print(" Done.")

    # 2. 生成背景场景 (使用 seed=0 的布局作为基准)
    base_target = generator.generate_target(scenario_type, dummy_xml, seed=0)
    
    # 3. 读取 XML 并注入所有目标点
    with open(dummy_xml, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # 移除原本生成的单个 target_viz (如果有的话)
    # 我们要手动添加 100 个
    
    # 构建 100 个小球的 XML 字符串
    ghost_bodies = []
    for i, pos in enumerate(targets):
        # 根据高度给一点颜色渐变，方便区分层次
        # 低处(垃圾桶)偏红，高处(柜子)偏绿
        r = max(0, min(1, 1.0 - pos[2]))
        g = max(0, min(1, pos[2]))
        rgba = f"{r} {g} 0.2 0.5"
        
        body_str = f"""
        <body name="ghost_target_{i}" pos="{pos[0]} {pos[1]} {pos[2]}">
            <geom type="sphere" size="0.02" rgba="{rgba}" contype="0" conaffinity="0"/>
        </body>
        """
        ghost_bodies.append(body_str)
    
    all_ghosts = "\n".join(ghost_bodies)
    
    # 插入到 </worldbody> 之前
    final_xml = xml_content.replace('</worldbody>', f'{all_ghosts}\n  </worldbody>')
    
    dist_xml_path = f"viz_distribution_{scenario_type}.xml"
    with open(dist_xml_path, 'w', encoding='utf-8') as f:
        f.write(final_xml)
    
    # 4. 启动 Viewer
    print(f"Launching Viewer for {scenario_type}. Close window to see next scenario.")
    model = mujoco.MjModel.from_xml_path(dist_xml_path)
    data = mujoco.MjData(model)
    
    # 设置一个好的初始视角
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 简单设置机器人位置避免挡视线
        data.qpos[0] = 0
        data.qpos[1] = 0
        mujoco.mj_forward(model, data)
        
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)

    # 清理临时文件
    if os.path.exists(dummy_xml): os.remove(dummy_xml)

if __name__ == "__main__":
    # 依次展示 4 个场景
    scenarios = ['living_room', 'kitchen', 'storage', 'corner']
    
    for s in scenarios:
        visualize_distribution(s, n_samples=200) # 生成 200 个点足够看清分布