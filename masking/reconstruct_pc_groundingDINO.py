import os
import glob
import json
import cv2
import torch
import numpy as np
import trimesh

########################
# PyTorch3D for transforms
########################
from pytorch3d.transforms import Transform3d

#########################################
# Segment Anything for masks (box -> mask)
#########################################
from segment_anything import sam_model_registry, SamPredictor

##################################################
# GroundingDINO for text-prompted box detection
##################################################
from groundingdino.util.inference import Model
from groundingdino.util.utils import load_model

# If you have a local clone of the repo:
#   from groundingdino.util.inference import load_image, predict
# otherwise adapt to your environment.

#############################################
# 1. Initialize GroundingDINO + SAM
#############################################

# GroundingDINO checkpoint
GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "groundingdino_swint_ogc.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GroundingDINO...")
dino_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG_PATH,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
)
dino_model = dino_model.to(device)


# Initialize SAM
SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"  # adjust your path
SAM_MODEL_TYPE = "vit_h"
print("Loading SAM model...")
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=device)

# We'll use SamPredictor for bounding box -> mask
sam_predictor = SamPredictor(sam)

######################################
# 2. HELPER FUNCTIONS (loading & utils)
######################################

def load_metadata(metadata_path):
    """
    Example structure of metadata.json with camera intrinsics & extrinsics.
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
    """
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    return depth


######################################
# 3. GroundingDINO: Detect bounding box
######################################

def detect_box_with_groundingdino(rgb_image, prompt="held by hands", box_threshold=0.3, text_threshold=0.25):
    """
    Run the GroundingDINO model on the given RGB image with the specified prompt.
    Returns an array of bounding boxes [x1, y1, x2, y2] in image coords.

    Args:
        prompt (str): The text prompt to ground, e.g., "held by hands"
        box_threshold (float): Confidence threshold for boxes
        text_threshold (float): Text threshold for matching
    """
    # Convert from RGB (H,W,3) to BGR if needed or process directly
    # GroundingDINO typically expects a PIL image or a numpy array in RGB.
    # We'll adapt to the example from the official code:

    # The 'Model' class from 'groundingdino.util.inference' can do the predictions:
    #   bounding_boxes, logits, phrases = dino_model.predict_with_caption(...)
    # or you might need to call a lower-level function.

    # For simplicity, let's define a quick example using the Model's forward function:
    # (Use the official code from the GroundingDINO repo for full usage.)

    model = Model(dino_model)  # wrapper with convenience methods
    # Convert from np.uint8 to PIL or just pass the array
    # bounding_boxes, confidence_scores, labels = model.predict(
    #     image=rgb_image,
    #     prompt=prompt,
    #     box_threshold=box_threshold,
    #     text_threshold=text_threshold
    # )

    # We'll mimic that usage:
    boxes, confidences, labels = model.predict(
        image=rgb_image,
        prompt=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    # boxes is Nx4 in XYXY format, confidences is Nx1, labels is Nx1
    # pick the highest confidence box or any box that has good confidence
    final_boxes = []
    for box, conf, label in zip(boxes, confidences, labels):
        if conf >= box_threshold:
            # box is [x1, y1, x2, y2]
            final_boxes.append(box.astype(np.int32))

    return final_boxes


###########################################
# 4. Convert Box -> Mask with SAM
###########################################

def box_to_mask_sam(rgb_image, boxes):
    """
    Given an RGB image and a list of bounding boxes,
    produce a single mask that best covers them.
    This is a simple example that:
      - runs SAM for each box
      - merges all resulting masks via logical OR
    """
    if len(boxes) == 0:
        # fallback: no boxes found, use entire image or empty
        h, w = rgb_image.shape[:2]
        return np.zeros((h, w), dtype=bool)
    
    sam_predictor.set_image(rgb_image)

    # We'll accumulate the masks in a single array
    # For each box, we do predictor.predict(...).
    final_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
    
    for box in boxes:
        # box must be in [x_min, y_min, x_max, y_max]
        box_arr = np.array([box])  # shape (1,4)
        masks, scores, _ = sam_predictor.predict(
            box=box_arr, 
            point_coords=None,
            point_labels=None,
            multimask_output=True
        )
        # pick best mask from scores
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]  # HxW bool
        final_mask = np.logical_or(final_mask, best_mask)

    return final_mask


############################################
# 5. Back-project to get a 3D point cloud
############################################

def backproject_to_pointcloud(rgb, depth, mask, intrinsics, extrinsics):
    """
    Convert masked pixels into 3D points.
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

    # Nx3 camera coords
    cam_points = np.stack((Xc, Yc, Zc), axis=1)
    colors = rgb[y_idxs, x_idxs, :]

    # Homogeneous coords Nx4
    ones = np.ones((cam_points.shape[0], 1), dtype=np.float32)
    cam_points_h = np.concatenate([cam_points, ones], axis=1)

    # Transform camera -> world
    world_points_h = cam_points_h @ extrinsics.T
    world_points = world_points_h[:, :3] / world_points_h[:, [3]]

    return world_points, colors


#################################
# 6. MERGE MULTI-VIEW POINT CLOUD
#################################

def process_multiview_folder(root_dir):
    """
    For each subfolder (0, 1, 2, ...):
      - load rgb/depth/metadata
      - detect bounding box with GroundingDINO prompt "held by hands"
      - convert bounding boxes to mask(s) with SAM
      - back-project to 3D
      - merge in a single point cloud
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

        # 2) Detect bounding box(es) with GroundingDINO
        boxes = detect_box_with_groundingdino(
            rgb_image=rgb, 
            prompt="held by hands",
            box_threshold=0.3,   # adjust as needed
            text_threshold=0.25  # adjust as needed
        )
        
        if len(boxes) == 0:
            print(f"[WARN] No boxes found for 'held by hands' in {folder}.")
            continue

        # 3) Convert bounding boxes to a single mask with SAM
        mask = box_to_mask_sam(rgb, boxes)

        # 4) Back-project masked region
        pts_3d, cols_3d = backproject_to_pointcloud(rgb, depth, mask, intrinsics, extrinsics)

        all_pts.append(pts_3d)
        all_cols.append(cols_3d)

    if len(all_pts) == 0:
        print("[ERROR] No valid data found in any subfolders.")
        return None, None

    merged_pts = np.concatenate(all_pts, axis=0)
    merged_cols = np.concatenate(all_cols, axis=0)

    # 5) (Optional) Save merged PLY
    pcl = trimesh.PointCloud(vertices=merged_pts, colors=merged_cols)
    out_path = os.path.join(root_dir, "merged_held_by_hands.ply")
    pcl.export(out_path)
    print(f"[INFO] Merged point cloud saved to: {out_path}")

    return merged_pts, merged_cols


############################
# 7. MAIN SCRIPT ENTRY POINT
############################

if __name__ == "__main__":
    root_dir = "./data/2025-02-06_20-55-50"  # your data path
    merged_pts, merged_cols = process_multiview_folder(root_dir)
    
    if merged_pts is None:
        print("[ERROR] No points merged. Exiting.")
        exit(0)

    print(f"[INFO] Merged cloud shape: {merged_pts.shape}")

    # (Optional) MinkowskiEngine for 3D feature extraction
    # ...

