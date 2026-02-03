import os
import trimesh
import numpy as np

class AssetsManager:
    def __init__(self, asset_dir="assets/ikea_processed"):
        self.asset_dir = asset_dir
        self.assets = {} 
        self._scan_assets()

    def _scan_assets(self):
        if not os.path.exists(self.asset_dir):
            return

        files = [f for f in os.listdir(self.asset_dir) if f.endswith('.obj')]
        print(f"[AssetsManager] Found {len(files)} IKEA models.")

        for f in files:
            name = os.path.splitext(f)[0]
            path = os.path.join(self.asset_dir, f)
            
            try:
                mesh = trimesh.load(path, process=False, force='mesh')
                # 原始 AABB
                extents = mesh.extents
                max_dim = np.max(extents)
                
                # 缩放逻辑
                scale_factor = 1.0
                if max_dim > 100:
                    scale_factor = 0.001
                elif max_dim > 10: 
                    scale_factor = 0.01
                
                real_extents = extents * scale_factor
                real_radius = np.linalg.norm(real_extents[:2]) / 2.0
                
                self.assets[name] = {
                    'filename': f,
                    # [关键修复] 统一使用 raw_extents 作为键名
                    'raw_extents': extents, 
                    'scale_factor': scale_factor,
                    'radius': real_radius
                }
                
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")

    def get_asset_xml_string(self):
        xml_str = "\n    <!-- IKEA Assets -->\n"
        for name, data in self.assets.items():
            rel_path = f"../../../assets/ikea_processed/{data['filename']}"
            s = data['scale_factor']
            xml_str += f'    <mesh name="mesh_{name}" file="{rel_path}" scale="{s} {s} {s}"/>\n'
        return xml_str