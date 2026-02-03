import numpy as np
import os
from src.env.assets_manager import AssetsManager

class SceneGenerator:
    def __init__(self, base_xml_path="configs/mobile_panda.xml"):
        self.base_xml_path = base_xml_path
        self.assets_mgr = AssetsManager()
        
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

    def _place(self, list_xml, key, pos, yaw):
        name, data = self._get_asset(key)
        if name:
            quat = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            xml = f'<body name="fix_{key}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"><geom type="mesh" mesh="mesh_{name}" material="off_white" contype="1" conaffinity="1" density="50" friction="0.8 0.005 0.0001"/></body>'
            list_xml.append(xml)

    def _sample_volume(self, center_pos, size_xyz):
        """
        在给定的长方体体积内均匀采样一个点。
        center_pos: 体积中心的全局坐标
        size_xyz: 体积的长宽高 [len_x, len_y, len_z]
        """
        half_extents = np.array(size_xyz) / 2.0
        noise = np.random.uniform(-half_extents, half_extents)
        return np.array(center_pos) + noise

    def generate_target(self, scenario_type, save_xml_path, seed=0):
        np.random.seed(seed)
        
        if not os.path.exists(self.base_xml_path):
            print(f"Error: Base XML {self.base_xml_path} not found.")
            return None
            
        with open(self.base_xml_path, 'r', encoding='utf-8') as f:
            base_content = f.read()
        
        asset_str = self.assets_mgr.get_asset_xml_string()
        base_content = base_content.replace('</asset>', f'{asset_str}\n  </asset>')
        
        xml_bodies = []
        
        # 存储采样区域：列表中的每个元素为 tuple (center, size)
        # center: 区域的全局中心点
        # size: 区域的 [x长度, y长度, z高度]
        sampling_regions = []

        # 家具位置微调 (Global Jitter)
        j_furn = np.random.uniform(-0.05, 0.05, 2)

        if scenario_type == 'living_room':
            # --- 家具布局 ---
            # Lack 桌子 (旋转了 90度/1.57rad)
            table_pos = [0.75 + j_furn[0], 0 + j_furn[1], 0]
            # 沙发 (未旋转)
            sofa_pos = [0.05, 1.0, 0] 
            # 侧边圆桌 (未旋转)
            side_table_pos = [0.05, -0.8 + j_furn[1], 0]

            self._place(xml_bodies, 'lack', table_pos, 1.57)
            self._place(xml_bodies, 'strandmon', sofa_pos, 0)
            self._place(xml_bodies, 'lersta', [0.4, -0.8, 0], 2.5)
            self._place(xml_bodies, 'tanebro', side_table_pos, 0.5)

            # --- 定义采样区域 ---
            
            # 1. Lack 桌子表面全覆盖
            # Lack 原始尺寸约 0.9x0.55。因为旋转了90度，Global X 是 0.55，Global Y 是 0.9。
            # 高度约 0.45。采样范围 Z: 0.45 ~ 0.65
            sampling_regions.append((
                np.array(table_pos) + [-0.15, 0, 0.4],  # Center Z slightly above table
                [0.55, 1.2, 0.45]                   # Size [X, Y, Z]
            ))

            # 2. 沙发坐垫全覆盖
            # 坐垫区域很大，覆盖整个座面。
            # Z: 0.40 ~ 0.60
            sampling_regions.append((
                np.array(sofa_pos) + [0, -0.28, 0.50], 
                [0.50, 0.6, 0.20]                    # Width 1.0m to cover full seat
            ))

            # 3. 侧边圆桌全覆盖
            # 直径约 0.5m。
            sampling_regions.append((
                np.array(side_table_pos) + [0, 0, 0.65],
                [0.45, 0.45, 0.20]
            ))

        elif scenario_type == 'kitchen':
            # --- 家具布局 ---
            # 岛台 (Skruvby) - 旋转 -90度
            island_pos = [0.7 + j_furn[0], 0 + j_furn[1], 0]
            # 餐桌 (Vihals) - 未旋转
            dining_pos = [-0.25, 0.9 + j_furn[1], 0]

            self._place(xml_bodies, 'skruvby', island_pos, -1.57)
            self._place(xml_bodies, 'pot', [island_pos[0]-0.05, island_pos[1]-0.3, 0.9], 0)
            self._place(xml_bodies, 'vihals', dining_pos, 0)
            self._place(xml_bodies, 'buslatt', [-0.25, 0.7, 0], 3.14)

            # --- 定义采样区域 ---

            # 1. 岛台表面全覆盖
            sampling_regions.append((
                np.array(island_pos) + [0, 0, 0.95], 
                [0.35, 1.1, 0.30] 
            ))

            # 2. 岛台下方柜子全覆盖
            sampling_regions.append((
                np.array(island_pos) + [0, -0.38, 0.45],
                [0.35, 0.3, 0.5] 
            ))

            # 3. 餐桌表面全覆盖
            sampling_regions.append((
                np.array(dining_pos) + [0, 0, 0.80],
                [1.2, 0.8, 0.20]
            ))

            # 4. 餐桌下方全覆盖
            sampling_regions.append((
                np.array(dining_pos) + [0, 0, 0.4],
                [1.2, 0.6, 0.40]
            ))

        elif scenario_type == 'storage':
            # --- 家具布局 ---
            # Kallax (侧放 1.57rad)
            kallax_pos = [0.95 + j_furn[0], 0 + j_furn[1], 0]
            # Billy (斜放 2.8rad)
            billy_pos = [0.3 + j_furn[0], -0.9, 0]

            self._place(xml_bodies, 'kallax', kallax_pos, 1.57)
            self._place(xml_bodies, 'billy', billy_pos, 2.8)
            # 盒子放在柜子里，属于“障碍物”，目标点可以在它们里面
            self._place(xml_bodies, 'drona', [0.95, -0.2, 0.75], 0) 
            self._place(xml_bodies, 'tjena', [0.5, -0.9, 0.1], 2.8)

            # --- 定义采样区域 (体积填充) ---

            # 1. Kallax 整体体积填充
            # 覆盖上下两层格子。Z范围从 0.2 到 0.9
            # 因为是侧放，X轴方向是厚度(0.4)，Y轴方向是长度(1.47)
            sampling_regions.append((
                np.array(kallax_pos) + [0, 0.0, 0.7], # 中心高度 0.55
                [0.35, 0.9, 1.1]                         # 高度范围覆盖 0.2~0.9
            ))

            # 2. Billy 书架体积填充
            # Billy 是斜放的，简单的 AABB (轴对齐包围盒) 可能会包含一些无效区域，
            # 但题目说"被碰撞检测过滤掉也没关系"，所以直接画一个覆盖大致区域的大盒子。
            sampling_regions.append((
                np.array(billy_pos) + [0.05, 0, 0.70], 
                [0.7, 0.2, 1.0] # 很大的一个方块，覆盖书架区域
            ))

        elif scenario_type == 'corner':
            # --- 家具布局 ---
            desk_pos = [0.85 + j_furn[0], -0.1 + j_furn[1], 0]
            drawer_pos = [0.5, 0.75 + j_furn[1], 0] 
            bin_pos = [0.4 + j_furn[0], -0.65, 0]   

            self._place(xml_bodies, 'micke', desk_pos, -1.57)
            self._place(xml_bodies, 'hemnes', drawer_pos, -1.57)
            self._place(xml_bodies, 'fniss', bin_pos, 0)
            self._place(xml_bodies, 'hektar', [0.9, 0.15, 0.76], -1.0)

            # --- 定义采样区域 ---

            # 1. 书桌表面
            sampling_regions.append((
                np.array(desk_pos) + [0, 0, 0.5],
                [0.5, 1.0, 0.8]
            ))

            # 2. 抽屉柜顶部
            sampling_regions.append((
                np.array(drawer_pos) + [0, 0, 0.75],
                [0.5, 0.5, 0.2]
            ))

            # 3. 垃圾桶内部全填充
            # 桶高约 0.3m。Z范围 0.05 ~ 0.35
            sampling_regions.append((
                np.array(bin_pos) + [0, 0, 0.20], # 中心 Z=0.2
                [0.20, 0.20, 0.30]                # 覆盖整个桶内体积
            ))

        # --- 执行采样 ---
        # 1. 随机选择一个区域 (Region)
        if not sampling_regions:
            final_target_pos = np.array([0.5, 0, 0.5])
        else:
            region_idx = np.random.choice(len(sampling_regions))
            center, size = sampling_regions[region_idx]
            
            # 2. 在该区域内进行体积采样 (Volume Sampling)
            final_target_pos = self._sample_volume(center, size)

        # 写入文件
        body_content = "\n".join(xml_bodies)
        target_viz = f"""
        <body name="target_viz" pos="{final_target_pos[0]} {final_target_pos[1]} {final_target_pos[2]}" mocap="true">
            <geom type="sphere" size="0.03" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
        </body>
        """
        final_xml = base_content.replace('</worldbody>', f'{body_content}\n{target_viz}\n  </worldbody>')
        
        with open(save_xml_path, 'w', encoding='utf-8') as f:
            f.write(final_xml)
            
        return final_target_pos