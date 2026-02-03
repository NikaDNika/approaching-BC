import trimesh
import os
import glob

# 检查前 3 个文件
files = glob.glob("assets/ikea_processed/*.obj")[:3]

print("=== Mesh Size Diagnosis ===")
for f in files:
    mesh = trimesh.load(f, process=False)
    print(f"File: {os.path.basename(f)}")
    print(f"  Extents (Size): {mesh.extents}")
    print(f"  Bounds (Min/Max): \n{mesh.bounds}")
    print("-" * 20)