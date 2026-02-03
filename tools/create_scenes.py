import os
import sys
import numpy as np

# 1. 动态添加项目根目录到 sys.path
# 当前文件: tools/create_scenes.py
# 根目录: DRP_FULL/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 导入模块
from src.env.assets_manager import AssetsManager

class FixedSceneCreator:
    def __init__(self, base_xml_name="mobile_panda.xml"):
        # 定义路径
        self.config_dir = os.path.join(project_root, "configs")
        self.base_xml_path = os.path.join(self.config_dir, base_xml_name)
        
        self.assets_mgr = AssetsManager()
        
        # 家具映射表
        self.file_map = {
            'lack': 'lack_coffee_table___black_brown_00104291',
            'vihals': 'vihals_table___whitewhite_39578509',
            'buslatt': 'buslätt_chair___whitepine_90601139',
            'micke': 'micke_desk___white_80213074',
            'tanebro': 'tånebro_side_table___indooroutdooranthra',
            'kallax': 'kallax_shelf_unit___white_80275887',
            'billy': 'billy_bookcase___oak_effect_90477385',
            'vittsjo': 'vittsjö_shelf_unit___white_10305802',
            'skruvby': 'skruvby_sideboard___black_blue_70568720',
            'hemnes': 'hemnes_2_drawer_chest___white_stain_8024',
            'strandmon': 'strandmon_wing_chair___nordvalla_dark_gr',
            'skadis': 'skådis_pegboard_combination_89406365',
            'lersta': 'lersta_floorreading_lamp___aluminum_chro',
            'hektar': 'hektar_work_lamp___dark_gray_90349374',
            'fniss': 'fniss_trash_can___white_40295439',
            'pot': 'ikea_365+_pot_with_lid___stainless_steel',
            'tjena': 'tjena_storage_box_with_lid___white_black',
            'drona': 'dröna_box___white_50467067'
        }

    def _get_asset(self, key):
        full_name = self.file_map.get(key)
        if not full_name: return None, None
        if full_name in self.assets_mgr.assets:
            return full_name, self.assets_mgr.assets[full_name]
        for loaded_name in self.assets_mgr.assets.keys():
            if full_name[:15] in loaded_name: return loaded_name, self.assets_mgr.assets[loaded_name]
        return None, None

    def create_scene(self, scenario_type):
        print(f"Creating Fixed Scene: {scenario_type}...")
        
        # 检查基础文件是否存在
        if not os.path.exists(self.base_xml_path):
            print(f"Error: Base XML not found at {self.base_xml_path}")
            return

        with open(self.base_xml_path, 'r', encoding='utf-8') as f:
            base_content = f.read()
        
        asset_str = self.assets_mgr.get_asset_xml_string()
        base_content = base_content.replace('</asset>', f'{asset_str}\n  </asset>')
        
        assets_abs_path = os.path.join(project_root, "mujoco_menagerie", "franka_emika_panda", "assets")

        if 'meshdir="' in base_content:
            import re
            # 替换 meshdir="任意内容" 为 meshdir="绝对路径"
            base_content = re.sub(r'meshdir="[^"]+"', f'meshdir="{assets_abs_path}"', base_content)

        xml_bodies = []
        
        # === 布局逻辑 (保持不变) ===
        if scenario_type == 'living_room':
            self._place(xml_bodies, 'lack', [0.75, 0, 0], 1.57)
            self._place(xml_bodies, 'strandmon', [0.05, 1, 0], 0)
            self._place(xml_bodies, 'lersta', [0.4, -0.8, 0], 2.5)
            self._place(xml_bodies, 'tanebro', [0.05, -0.8, 0], 0.5)

        elif scenario_type == 'kitchen':
            self._place(xml_bodies, 'skruvby', [0.7, 0, 0], -1.57)
            self._place(xml_bodies, 'pot', [0.65, -0.3, 0.9], 0)
            self._place(xml_bodies, 'vihals', [-0.25, 0.9, 0], 0)
            self._place(xml_bodies, 'buslatt', [-0.25, 0.7, 0], 3.14)

        elif scenario_type == 'storage':
            self._place(xml_bodies, 'kallax', [0.95, 0, 0], 1.57)
            self._place(xml_bodies, 'billy', [0.3, -0.9, 0], 2.8)
            self._place(xml_bodies, 'drona', [0.95, -0.2, 0.75], 0)
            self._place(xml_bodies, 'tjena', [0.5, -0.9, 0.1], 2.8)

        elif scenario_type == 'corner':
            self._place(xml_bodies, 'micke', [0.85, -0.1, 0], -1.57)
            self._place(xml_bodies, 'hemnes', [0.5, 0.75, 0], -1.57)
            self._place(xml_bodies, 'fniss', [0.4, -0.65, 0], 0)
            self._place(xml_bodies, 'hektar', [0.9, 0.15, 0.76], -1.0)

        body_content = "\n".join(xml_bodies)
        final_xml = base_content.replace('</worldbody>', f'{body_content}\n    <!-- TARGET_PLACEHOLDER -->\n  </worldbody>')
        
        # [修改] 保存路径也改为 configs/ 目录下
        filename = os.path.join(self.config_dir, f"fixed_scene_{scenario_type}.xml")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_xml)
        print(f"Saved to {filename}")

    def _place(self, list_xml, key, pos, yaw):
        name, data = self._get_asset(key)
        if name:
            quat = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            xml = f'<body name="fix_{key}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"><geom type="mesh" mesh="mesh_{name}" material="off_white" contype="1" conaffinity="1" density="50" friction="0.8 0.005 0.0001"/></body>'
            list_xml.append(xml)

if __name__ == "__main__":
    creator = FixedSceneCreator()
    for s in ['living_room', 'kitchen', 'storage', 'corner']:
        creator.create_scene(s)