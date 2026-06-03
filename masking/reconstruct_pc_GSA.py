import os
import glob
import json
import cv2
import torch
import numpy as np
import trimesh
from PIL import Image

# from your local files:
from GroundSamWraper import GroundSam_Wraper  # or wherever your GroundSam_Wraper is defined
from IPython import embed

#######################################################
# 1) Camera & Depth Loading (same as your prior script)
#######################################################

def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    fx = meta["camera_data"]["fx"]
    fy = meta["camera_data"]["fy"]
    cx = meta["camera_data"]["cx"]
    cy = meta["camera_data"]["cy"]

    intrinsics = np.array([
        [fx,   0,  cx],
        [0,   fy,  cy],
        [0,    0,   1],
    ], dtype=np.float32)

    pose = meta["camera_data"]["camera_pose"]
    origin = np.array(pose["origin"], dtype=np.float32)
    x_axis = np.array(pose["x"], dtype=np.float32)
    y_axis = np.array(pose["y"], dtype=np.float32)
    z_axis = np.array(pose["z"], dtype=np.float32)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (3,3)
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = origin

    return intrinsics, extrinsics

def load_rgb_image(rgb_path):
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

#def load_depth_image(depth_path):
#    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
#    return depth
def load_depth_image(depth_path, target_shape=None):
    """
    If target_shape = (H, W), we resize the depth to that shape using NEAREST interpolation.
    """
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if target_shape is not None:
        depth_resized = cv2.resize(depth, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return depth_resized
    return depth

#######################################
# 2) Back-Projection (unchanged)
#######################################

#def backproject_to_pointcloud(rgb, depth, mask, intrinsics, extrinsics, depth_scale=0.001):
def backproject_to_pointcloud(rgb, depth, mask, intrinsics, extrinsics, depth_scale=1.0):
    """
    - rgb:     H x W x 3  (np.uint8)
    - depth:   H x W      (np.uint16 or float)
    - mask:    H x W      (bool)
    - intrinsics: 3x3
    - extrinsics: 4x4
    - depth_scale: optional factor if your depth is in mm (e.g., 0.001)
    """
    # 1) Find all masked pixels
    y_idxs, x_idxs = np.where(mask)
    print(f"[DEBUG] Masked pixels: {len(y_idxs)}")

    # 2) Clamp or filter out-of-bounds
    H, W = depth.shape[:2]
    in_bounds = (y_idxs < H) & (x_idxs < W)
    y_idxs = y_idxs[in_bounds]
    x_idxs = x_idxs[in_bounds]
    print(f"[DEBUG] In-bounds pixels: {len(y_idxs)}")

    # 3) Extract and scale depth
    #    If your depth is 16-bit in millimeters, set depth_scale=0.001
    z_vals = depth[y_idxs, x_idxs].astype(np.float32) * depth_scale

    # 4) Filter out zero or invalid depths
    valid_depth = (z_vals > 0)
    y_idxs = y_idxs[valid_depth]
    x_idxs = x_idxs[valid_depth]
    z_vals = z_vals[valid_depth]
    print(f"[DEBUG] Non-zero depth pixels: {len(y_idxs)}")

    # 5) Project to camera coords
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    Xc = (x_idxs - cx) * z_vals / fx
    Yc = (y_idxs - cy) * z_vals / fy
    Zc = z_vals

    cam_points = np.stack((Xc, Yc, Zc), axis=1)

    # 6) Get corresponding colors
    colors = rgb[y_idxs, x_idxs, :]

    # 7) Convert to homogeneous coords
    ones = np.ones((cam_points.shape[0], 1), dtype=np.float32)
    cam_points_h = np.concatenate([cam_points, ones], axis=1)

    # 8) Transform camera -> world
    world_points_h = cam_points_h @ extrinsics.T
    world_points = world_points_h[:, :3] / world_points_h[:, [3]]

    print(f"[DEBUG] Final 3D points: {world_points.shape[0]}")
    return world_points, colors


#####################################################
# 3) Process Multi-View with GroundSam_Wraper
#####################################################

def process_multiview_folder(root_dir, ground_sam: GroundSam_Wraper, text_prompt="held by hands"):
    subfolders = sorted(glob.glob(os.path.join(root_dir, "*")))

    all_pts = []
    all_cols = []

    for folder in subfolders:
        if not os.path.isdir(folder):
            continue

        rgb_path = os.path.join(folder, "rgb_image.png")
        depth_path = os.path.join(folder, "depth_image.png")
        meta_path = os.path.join(folder, "metadata.json")

        if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(meta_path)):
            print(f"Skipping {folder}; missing files.")
            continue

        # 1) Load data
        rgb = load_rgb_image(rgb_path)
        #depth = load_depth_image(depth_path)
        depth = load_depth_image(depth_path, target_shape=rgb.shape[:2])  # shape becomes (960, 720)

        intrinsics, extrinsics = load_metadata(meta_path)

        # 2) Prepare the image for GroundingDINO (just convert to tensor/normalize)
        from torchvision import transforms as tvt
        transform = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pil_image = Image.fromarray(rgb)  # shape (H, W, 3)
        image_tensor = transform(pil_image).to(ground_sam.device)  # shape (3, H, W)

        # 3) Grounding DINO bounding boxes for text prompt
        boxes_filt, pred_phrases = ground_sam.text_dino_grounding(
            image_tensor,
            det_prompt=text_prompt,
            box_threshold=0.5,   # tweak as needed
            text_threshold=0.10  # tweak as needed
        )

        # 4) Use bounding boxes to get SAM masks
        H_img, W_img = rgb.shape[:2]
        try:
            masks, _ = ground_sam.sam_bbox(
                image_path=rgb_path,
                image_size=(W_img, H_img),
                boxes_filt=boxes_filt
            )
        except:
            continue

        if masks is None or len(masks) == 0:
            print(f"No mask found in folder {folder}. Skipping.")
            continue

        # Combine multiple masks with OR
        combined_mask = np.zeros((H_img, W_img), dtype=bool)
        for i in range(masks.shape[0]):
            mask_i = masks[i]
            # If mask_i is [1, H, W], remove the extra dim
            if mask_i.ndim == 3:
                mask_i = mask_i[0]  # shape [H, W]
            mask_i = mask_i.cpu().numpy().astype(bool)
            combined_mask = np.logical_or(combined_mask, mask_i)

        ############
        # 5A) SAVE MASK
        ############
        # Example: save the binary mask as a PNG (0=background, 255=foreground)
        mask_save_path = os.path.join(folder, f"mask_{text_prompt.replace(' ', '_')}.png")
        mask_255 = (combined_mask * 255).astype(np.uint8)
        cv2.imwrite(mask_save_path, mask_255)
        print(f"[INFO] Saved mask to: {mask_save_path}")

        ############
        # 5B) OPTIONAL: OVERLAY MASK ON RGB FOR VISUAL CHECK
        ############
        # This creates a color overlay on the original image
        overlay = rgb.copy()
        overlay[combined_mask] = [255, 0, 0]  # e.g. highlight in red
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_save_path = os.path.join(folder, f"overlay_{text_prompt.replace(' ', '_')}.png")
        cv2.imwrite(overlay_save_path, overlay_bgr)
        print(f"[INFO] Saved overlay to: {overlay_save_path}")

        # 6) Back-project to get 3D points
        pts_3d, cols_3d = backproject_to_pointcloud(rgb, depth, combined_mask, intrinsics, extrinsics)
        all_pts.append(pts_3d)
        all_cols.append(cols_3d)

    if len(all_pts) == 0:
        print("No valid data found or no masks in any subfolders.")
        return None, None

    merged_pts = np.concatenate(all_pts, axis=0)
    merged_cols = np.concatenate(all_cols, axis=0)

    # Save merged PLY
    out_path = os.path.join(root_dir, f"merged_{text_prompt.replace(' ', '_')}.ply")
    pcl = trimesh.PointCloud(vertices=merged_pts, colors=merged_cols)
    pcl.export(out_path)
    print(f"[INFO] Merged point cloud saved to: {out_path}")

    return merged_pts, merged_cols

###############################################################
# 4) MAIN
###############################################################

if __name__ == "__main__":
    root_dir = "./data/2025-02-06_20-55-50"

    # Example usage
    DINO_CONFIG_FILE = "./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    DINO_CHECKPOINT = "./Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
    SAM_CHECKPOINT = "./Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

    ground_sam = GroundSam_Wraper(
        config_file=DINO_CONFIG_FILE,
        grounded_checkpoint=DINO_CHECKPOINT,
        sam_checkpoint=SAM_CHECKPOINT,
        device="cuda"  # or "cpu"
    )

    text_prompt = "the center object held by hands"
    merged_pts, merged_cols = process_multiview_folder(root_dir, ground_sam, text_prompt=text_prompt)

    if merged_pts is not None:
        print("[INFO] Final merged cloud shape:", merged_pts.shape)
    else:
        print("[ERROR] No final merged data. Exiting.")

