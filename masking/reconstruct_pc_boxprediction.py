import os
import glob
import json
import cv2
import torch
import numpy as np
import trimesh

#############################
# PyTorch3D (for transforms)
#############################
from pytorch3d.transforms import Transform3d

##################################################
# Segment Anything (for bounding box or point promps)
##################################################
from segment_anything import sam_model_registry, SamPredictor

##########################################################
# (Optional) MinkowskiEngine for 3D feature extraction
##########################################################
#import MinkowskiEngine as ME
import torch.nn as nn


#############################################
# 1. Initialize Segment Anything (SAM) Model
#############################################

# Path to your SAM checkpoint. Adjust if needed.
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"  # or "vit_l", "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading SAM model...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=device)

# Instead of mask_generator, we use SamPredictor
predictor = SamPredictor(sam)


######################################
# 2. HELPER FUNCTIONS (loading & utils)
######################################

def load_metadata(metadata_path):
    """
    Example metadata structure:

    {
      "timestamp": "2025-02-06_20-55-50",
      ...
      "camera_data": {
        "fx": 197.16957,
        "fy": 197.16957,
        "cx": 128.16407,
        "cy": 96.29129,
        "camera_pose": {
          "origin": [0.00079, -0.00245, -0.00036],
          "x": [0.0306379, -0.6951289, 0.7182318],
          "y": [0.9995303, 0.0209879, -0.0223245],
          "z": [0.0004442, 0.7185785, 0.6954455]
        }
      }
    }
    We'll construct:
      - intrinsics = [[fx, 0,  cx],
                      [0,  fy, cy],
                      [0,   0,  1 ]]
      - extrinsics = 4x4 from camera_pose
          R = [x y z] as columns
          t = origin
    """

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    # Parse intrinsics
    fx = meta["camera_data"]["fx"]
    fy = meta["camera_data"]["fy"]
    cx = meta["camera_data"]["cx"]
    cy = meta["camera_data"]["cy"]

    intrinsics = np.array([
        [fx,   0,  cx],
        [0,   fy,  cy],
        [0,    0,   1],
    ], dtype=np.float32)

    # Parse extrinsics from camera_pose
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
    """Load the RGB image using OpenCV (BGR) and convert to RGB."""
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def load_depth_image(depth_path):
    """
    Load the depth map. If your depth is 16-bit or float, handle accordingly.
    Adjust scaling if necessary.
    """
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # For example, if 16-bit:
    # depth = depth.astype(np.float32) * 0.001
    return depth


####################################
# 3. SAM Prompting for the "object"
####################################

def segment_object_held_by_hands(rgb_image, box=None):
    """
    Prompt SAM with a bounding box (or points) for the object.
    If no box is provided, we can define a default box
    (e.g. center region) or prompt the user.

    NOTE: For a *handheld object*, you might have a known region
          or a rough bounding box. This is just an example.
    """
    predictor.set_image(rgb_image)

    # If no bounding box was provided, let's guess a big center box.
    # Alternatively, you could define a manual bounding box for each image.
    H, W = rgb_image.shape[:2]
    if box is None:
        # For example, pick a bounding box around the center half of the image
        # [x_min, y_min, x_max, y_max]
        x_min, y_min = int(W*0.25), int(H*0.25)
        x_max, y_max = int(W*0.75), int(H*0.75)
        box = np.array([x_min, y_min, x_max, y_max])

    # SAM requires the box in [x_min, y_min, x_max, y_max] format
    # then we do a forward pass
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],  # Must be 2D array of shape (B, 4)
        multimask_output=True
    )

    # We can pick the highest scored mask
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]  # HxW boolean

    return best_mask


############################################
# 4. Back-projection to get a 3D point cloud
############################################

def backproject_to_pointcloud(rgb, depth, mask, intrinsics, extrinsics):
    """
    Back-project the masked pixels (u, v, depth) into 3D, using:
      Xc = (u - cx) * Z / fx
      Yc = (v - cy) * Z / fy
      Zc = Z
    Then apply extrinsics (4x4) to get world coordinates.
    Returns:
      - world_points (Nx3)
      - colors (Nx3) in RGB
    """
    # Find all masked pixels
    y_idxs, x_idxs = np.where(mask)

    # Clamp or filter out-of-bounds indices
    H, W = depth.shape[:2]
    in_bounds = (y_idxs < H) & (x_idxs < W)
    y_idxs = y_idxs[in_bounds]
    x_idxs = x_idxs[in_bounds]

    # Grab depth values
    z_vals = depth[y_idxs, x_idxs].astype(np.float32)

    # Filter out zero or invalid depths if needed
    valid_depth = (z_vals > 0)
    y_idxs = y_idxs[valid_depth]
    x_idxs = x_idxs[valid_depth]
    z_vals = z_vals[valid_depth]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    Xc = (x_idxs - cx) * z_vals / fx
    Yc = (y_idxs - cy) * z_vals / fy
    Zc = z_vals

    # Camera coords Nx3
    cam_points = np.stack((Xc, Yc, Zc), axis=1)

    # Colors Nx3 in RGB
    colors = rgb[y_idxs, x_idxs, :]

    # Homogeneous coords Nx4
    ones = np.ones((cam_points.shape[0], 1), dtype=np.float32)
    cam_points_h = np.concatenate([cam_points, ones], axis=1)

    # Transform camera -> world
    world_points_h = cam_points_h @ extrinsics.T
    world_points = world_points_h[:, :3] / world_points_h[:, [3]]

    return world_points, colors


#################################
# 5. MERGE MULTI-VIEW POINT CLOUD
#################################

def process_multiview_folder(root_dir):
    """
    For each subfolder (0, 1, 2, ...), load rgb/depth/metadata,
    segment with SAM (using a bounding box prompt),
    back-project to 3D, and accumulate in a big point cloud.
    """
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
        depth = load_depth_image(depth_path)
        intrinsics, extrinsics = load_metadata(meta_path)

        # 2) Segment the object with a bounding box or points
        #    If you have a custom box for each image, pass it into the function.
        mask = segment_object_held_by_hands(rgb, box=None)

        # 3) Back-project masked region
        pts_3d, cols_3d = backproject_to_pointcloud(rgb, depth, mask, intrinsics, extrinsics)

        all_pts.append(pts_3d)
        all_cols.append(cols_3d)

    if len(all_pts) == 0:
        print("No valid data found in any subfolders.")
        return None, None

    merged_pts = np.concatenate(all_pts, axis=0)
    merged_cols = np.concatenate(all_cols, axis=0)

    # 4) Save merged PLY (optional)
    pcl = trimesh.PointCloud(vertices=merged_pts, colors=merged_cols)
    out_path = os.path.join(root_dir, "merged_center_object.ply")
    pcl.export(out_path)
    print(f"[INFO] Merged point cloud saved to: {out_path}")

    return merged_pts, merged_cols


###########################################################
# 6. (Optional) 3D FEATURE EXTRACTION WITH MINKOWSKIENGINE
###########################################################
# For brevity, we won't redefine it here unless needed.


############################
# 7. MAIN SCRIPT ENTRY POINT
############################

if __name__ == "__main__":
    root_dir = "./data/2025-02-06_20-55-50"  # Update to your real path

    merged_pts, merged_cols = process_multiview_folder(root_dir)
    if merged_pts is None:
        print("[ERROR] No points merged. Exiting.")
        exit(0)

    print(f"[INFO] Merged cloud shape: {merged_pts.shape}")

    # If you'd like to proceed with 3D feature extraction, restore MinkowskiEngine code here:
    # model_3d = ExampleMinkUNet(in_nchannel=3, out_nchannel=32, D=3).to(device)
    # ...

