import os
import subprocess
import glob
import sys

# --- 配置 ---
# 你的 Blender 路径
BLENDER_EXE = "blender" 
# 如果 blender 命令不在 PATH 中，请解除下行注释并填入 /snap/bin/blender
# BLENDER_EXE = "/snap/bin/blender"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "assets", "ikea_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "assets", "ikea_processed")

# Blender 内部脚本
BLENDER_SCRIPT = """
import bpy
import os
import sys

# 获取传入的参数
argv = sys.argv
try:
    idx = argv.index("--")
    input_path = argv[idx + 1]
    output_path = argv[idx + 2]
except ValueError:
    print("ERROR: Arguments not found after '--'")
    sys.exit(1)

print(f"Processing: {input_path}")

# 1. 清空场景
bpy.ops.wm.read_factory_settings(use_empty=True)

try:
    # 2. 导入 GLB
    bpy.ops.import_scene.gltf(filepath=input_path)

    # 3. 选中所有 Mesh
    bpy.ops.object.select_all(action='DESELECT')
    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not mesh_objs:
        print("No mesh objects found!")
    else:
        # 4. 预处理：缩放和归一化
        # IKEA 模型通常是毫米单位，MuJoCo 需要米。缩小 0.001 倍。
        # 同时将原点移到底部中心，方便放置。
        
        # 选中所有
        for obj in mesh_objs:
            obj.select_set(True)
        
        bpy.context.view_layer.objects.active = mesh_objs[0]
        
        # 缩放 (0.001)
        # bpy.ops.transform.resize(value=(0.001, 0.001, 0.001))
        bpy.ops.transform.resize(value=(1, 1, 1))
        
        # 应用缩放 (Apply Scale) - 非常重要，否则导出时可能还是大的
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # 5. 导出 OBJ
        # 适配 Blender 4.0/5.0+ 的新 API
        if hasattr(bpy.ops.wm, "obj_export"):
            print("Using New API (wm.obj_export)")
            bpy.ops.wm.obj_export(
                filepath=output_path,
                export_selected_objects=True,
                export_materials=False, # MuJoCo 不需要 mtl
                forward_axis='Y',       # MuJoCo 坐标系适配
                up_axis='Z'
            )
        # 兼容旧版 Blender (< 4.0)
        elif hasattr(bpy.ops.export_scene, "obj"):
            print("Using Legacy API (export_scene.obj)")
            bpy.ops.export_scene.obj(
                filepath=output_path,
                use_selection=True,
                use_materials=False,
                axis_forward='Y',
                axis_up='Z'
            )
        else:
            print("Error: No suitable OBJ exporter found!")
            sys.exit(1)
            
        print(f"Successfully exported to {output_path}")

except Exception as e:
    print(f"Blender Internal Error: {e}")
    sys.exit(1)
"""

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    script_path = os.path.join(BASE_DIR, "temp_blender_script.py")
    with open(script_path, "w") as f:
        f.write(BLENDER_SCRIPT)
        
    glb_files = glob.glob(os.path.join(INPUT_DIR, "*.glb"))
    print(f"Found {len(glb_files)} GLB files in {INPUT_DIR}")
    
    success_count = 0
    
    for i, glb in enumerate(glb_files):
        name = os.path.splitext(os.path.basename(glb))[0]
        # 文件名清理：替换空格，转小写
        safe_name = name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()
        # 截断过长文件名
        if len(safe_name) > 40: safe_name = safe_name[:40]
        
        obj_name = f"{safe_name}.obj"
        output_path = os.path.join(OUTPUT_DIR, obj_name)
        
        print(f"[{i+1}/{len(glb_files)}] Converting {name} ...")
        
        cmd = [
            BLENDER_EXE,
            "--background",
            "--python", script_path,
            "--",
            glb,
            output_path
        ]
        
        # 运行并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed: {name}")
            # 只打印最后几行错误信息，避免刷屏
            err_lines = result.stdout.split('\n')[-10:]
            for line in err_lines:
                if line.strip(): print(f"   {line}")
        else:
            if os.path.exists(output_path):
                print(f"✅ Success: {obj_name}")
                success_count += 1
            else:
                print(f"⚠️ Unknown Error: Blender finished but file missing.")
    
    if os.path.exists(script_path):
        os.remove(script_path)
        
    print(f"\nAll Done! {success_count}/{len(glb_files)} converted.")
    print(f"Check directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()