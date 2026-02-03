import numpy as np
import cv2
import os
import glob
import time

def normalize_depth(depth_map):
    valid_mask = (depth_map > 0) & (depth_map < 3.0)
    if np.sum(valid_mask) == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    min_d = np.min(depth_map[valid_mask])
    max_d = np.max(depth_map[valid_mask])
    norm_depth = (depth_map - min_d) / (max_d - min_d + 1e-6)
    norm_depth = np.clip(norm_depth, 0, 1)
    norm_depth_uint8 = (norm_depth * 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(norm_depth_uint8, cv2.COLORMAP_JET)
    return colored_depth

def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def resize_if_needed(img, target_h, target_w):
    if img.shape[:2] != (target_h, target_w):
        return cv2.resize(img, (target_w, target_h))
    return img

def inspect_file(npz_file):
    print(f"\n{'='*20}\nInspecting: {npz_file}\n{'='*20}")
    
    try:
        data = np.load(npz_file)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    # --- 1. 检查键和形状 ---
    required_keys = [
        'cam_left_rgb', 'cam_left_depth',
        'cam_right_rgb', 'cam_right_depth',
        'cam_wrist_rgb', 'cam_wrist_depth'
    ]
    
    for k in required_keys:
        if k not in data:
            print(f"MISSING key: {k}")
            return

    n_frames = data['cam_left_rgb'].shape[0]
    print(f"Frames: {n_frames}")
    print("Press 'q' to stop playback early.")

    # 存储视频帧
    video_frames = []

    # --- 2. 播放循环 ---
    for t in range(n_frames):
        try:
            rgb_l = data['cam_left_rgb'][t]
            dep_l = data['cam_left_depth'][t]
            rgb_r = data['cam_right_rgb'][t]
            dep_r = data['cam_right_depth'][t]
            rgb_w = data['cam_wrist_rgb'][t]
            dep_w = data['cam_wrist_depth'][t]
            
            bgr_l = to_bgr(rgb_l)
            bgr_r = to_bgr(rgb_r)
            bgr_w = to_bgr(rgb_w)
            
            vis_dep_l = normalize_depth(dep_l)
            vis_dep_r = normalize_depth(dep_r)
            vis_dep_w = normalize_depth(dep_w)
            
            # 统一尺寸
            h, w = bgr_l.shape[:2]
            vis_dep_l = resize_if_needed(vis_dep_l, h, w)
            bgr_r = resize_if_needed(bgr_r, h, w)
            vis_dep_r = resize_if_needed(vis_dep_r, h, w)
            bgr_w = resize_if_needed(bgr_w, h, w)
            vis_dep_w = resize_if_needed(vis_dep_w, h, w)
            
            # 加标签
            def draw_label(img, text):
                cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            draw_label(bgr_l, "Left RGB")
            draw_label(vis_dep_l, "Left Depth")
            draw_label(bgr_r, "Right RGB")
            draw_label(vis_dep_r, "Right Depth")
            draw_label(bgr_w, "Wrist RGB")
            draw_label(vis_dep_w, "Wrist Depth")

            # 拼接
            row1 = np.hstack([bgr_l, vis_dep_l])
            row2 = np.hstack([bgr_r, vis_dep_r])
            row3 = np.hstack([bgr_w, vis_dep_w])
            grid = np.vstack([row1, row2, row3])
            
            # 缩放 (太大的话)
            if grid.shape[0] > 1000:
                scale = 0.8
                grid = cv2.resize(grid, (int(grid.shape[1]*scale), int(grid.shape[0]*scale)))
            
            cv2.imshow("Multimodal Data Inspection", grid)
            video_frames.append(grid)
            
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        
        except Exception as e:
            print(f"Error at frame {t}: {e}")
            break
            
    cv2.destroyAllWindows()
    
    # --- 3. 询问保存 ---
    if not video_frames: return

    user_input = input("\nSave visualization to video? (y/n): ").strip().lower()
    if user_input == 'y':
        # 获取视频尺寸
        h, w = video_frames[0].shape[:2]
        # 文件名: viz_原文件名.mp4
        base_name = os.path.basename(npz_file).replace('.npz', '.mp4')
        out_path = f"viz_{base_name}"
        
        # 初始化 Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或 'XVID'
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
        
        print(f"Saving {len(video_frames)} frames to {out_path}...")
        for f in video_frames:
            out.write(f)
        out.release()
        print("Done.")
    else:
        print("Skipped saving.")

def main():
    data_dir = "/media/j16/8deb00a4-cceb-e842-a899-55532424da473/dataset_v2"
    files = glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)
    
    if not files:
        print("No files found!")
        return
        
    files.sort(key=os.path.getmtime, reverse=True)
    inspect_file(files[0])

if __name__ == "__main__":
    main()