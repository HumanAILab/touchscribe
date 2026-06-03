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

#################################################
# Segment Anything (for automatic object masks)
#################################################
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

######################################################
# MinkowskiEngine (for 3D feature extraction example)
######################################################
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
mask_generator = SamAutomaticMaskGenerator(sam)


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
    origin = np.array(pose["origin"], dtype=np.float32)  # shape (3,)
    # Each of x,y,z is a 3-vector representing the camera's coordinate axes
    x_axis = np.array(pose["x"], dtype=np.float32)
    y_axis = np.array(pose["y"], dtype=np.float32)
    z_axis = np.array(pose["z"], dtype=np.float32)

    # Construct rotation matrix. If x,y,z are columns, then:
    #    R = [ x_axis  y_axis  z_axis ] in column form
    # i.e. R[:,0] = x_axis, R[:,1] = y_axis, ...
    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (3,3)

    # Build extrinsic matrix [R | t; 0 0 0 1]
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
    # Example: If your depth was stored as 16-bit in millimeters, you might do:
    # depth = depth.astype(np.float32) * 0.001
    return depth


############################################
# 3. SEGMENTATION + BACK-PROJECTION PIPELINE
############################################

def segment_center_object_sam(rgb_image):
    """
    Use SAM to get one or more object masks from the image.
    We pick the largest mask as a naive "center object."
    """
    masks = mask_generator.generate(rgb_image)
    if len(masks) == 0:
        # fallback: no mask found, use entire image
        return np.ones(rgb_image.shape[:2], dtype=bool)
    
    best_mask = None
    best_area = 0
    for m in masks:
        segmentation = m["segmentation"]  # HxW boolean
        area = np.sum(segmentation)
        if area > best_area:
            best_area = area
            best_mask = segmentation
    
    return best_mask

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
# 4. MERGE MULTI-VIEW POINT CLOUD
#################################

def process_multiview_folder(root_dir):
    """
    For each subfolder (0, 1, 2, ...), load rgb/depth/metadata,
    segment with SAM, back-project to 3D, and accumulate in a big point cloud.
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
        
        # Load data
        rgb = load_rgb_image(rgb_path)
        depth = load_depth_image(depth_path)
        intrinsics, extrinsics = load_metadata(meta_path)

        # Segment to find center object
        mask = segment_center_object_sam(rgb)

        # Back-project masked region
        pts_3d, cols_3d = backproject_to_pointcloud(rgb, depth, mask, intrinsics, extrinsics)

        all_pts.append(pts_3d)
        all_cols.append(cols_3d)
    
    if len(all_pts) == 0:
        print("No valid data found in any subfolders.")
        return None, None
    
    merged_pts = np.concatenate(all_pts, axis=0)
    merged_cols = np.concatenate(all_cols, axis=0)

    # Save merged PLY (optional)
    pcl = trimesh.PointCloud(vertices=merged_pts, colors=merged_cols)
    out_path = os.path.join(root_dir, "merged_center_object.ply")
    pcl.export(out_path)
    print(f"[INFO] Merged point cloud saved to: {out_path}")

    return merged_pts, merged_cols


###########################################################
# 5. EXAMPLE: 3D FEATURE EXTRACTION WITH MINKOWSKIENGINE
###########################################################
# Here we show a simple MinkowskiEngine-based pipeline
# to extract features from the merged 3D point cloud.

#class ExampleMinkUNet(nn.Module):
#    """
#    A small example Minkowski UNet model.
#    (Adapted from MinkowskiEngine examples.)
#    """
#    def __init__(self, in_nchannel=3, out_nchannel=32, D=3):
#        super().__init__()
#        # A typical MinkUNet could be large; here we do a simpler illustration
#        # MinkowskiEngine has MinkowskiConvolution, MinkowskiBatchNorm, MinkowskiReLU, ...
#        # For brevity, let's define a very small net:
#        self.net = nn.Sequential(
#            ME.MinkowskiConvolution(
#                in_nchannel, 16, kernel_size=3, stride=1, dimension=D
#            ),
#            ME.MinkowskiBatchNorm(16),
#            ME.MinkowskiReLU(),
#
#            ME.MinkowskiConvolution(
#                16, out_nchannel, kernel_size=3, stride=1, dimension=D
#            ),
#            ME.MinkowskiGlobalAvgPooling(),
#        )
#
#    def forward(self, x: ME.SparseTensor):
#        """
#        x: MinkowskiEngine SparseTensor
#        """
#        out = self.net(x)
#        # out.F is the feature tensor (N_batch x out_nchannel)
#        # but here, because of global average pooling, shape is (batch_size, out_nchannel)
#        return out.F  # return the dense features
#
#
#def extract_3d_features_minkowski(pts_3d, cols_3d, model_3d, voxel_size=0.01):
#    """
#    Convert a Nx3 point cloud (pts_3d) + Nx3 colors (cols_3d) into MinkowskiEngine input,
#    then run a forward pass to get global 3D features.
#    Args:
#      pts_3d: (N, 3) float
#      cols_3d: (N, 3) uint8 or float
#      model_3d: a MinkowskiEngine model
#      voxel_size: how large each voxel is (in same units as pts_3d)
#    Returns:
#      features: (1, out_channels) global feature vector
#    """
#
#    # 1) Convert 3D coords to discrete voxel indices
#    # MinkowskiEngine requires integer coordinates for SparseTensor.
#    # We do floor division or rounding:
#    coords = pts_3d / voxel_size
#    coords = np.floor(coords).astype(np.int32)
#
#    # 2) Prepare features (e.g., normalized RGB).
#    # Minkowski supports any dimension of features. Let’s do normalized RGB in [0,1].
#    feats = (cols_3d / 255.0).astype(np.float32)
#
#    # MinkowskiEngine expects coords and feats as torch tensors
#    coords_t = torch.from_numpy(coords)
#    feats_t = torch.from_numpy(feats)
#
#    # 3) Create a SparseTensor. We assume a single "batch index" = 0.
#    # So we need to add a column of zeros to coords for batch indices:
#    batch_indices = np.zeros((coords.shape[0], 1), dtype=np.int32)
#    input_coords = np.hstack((batch_indices, coords))
#    input_coords_t = torch.from_numpy(input_coords)
#
#    # Build the SparseTensor
#    in_tensor = ME.SparseTensor(
#        coordinates=input_coords_t,
#        features=feats_t,
#        device=device
#    )
#
#    # 4) Run the model
#    model_3d.eval()
#    with torch.no_grad():
#        features = model_3d(in_tensor)  # (1, out_channels) after global pooling
#    return features


############################
# 6. MAIN SCRIPT ENTRY POINT
############################

if __name__ == "__main__":
    # 1) Load and merge multi-view data
    root_dir = "./data/2025-02-06_20-55-50"  # Adjust to your real path
    merged_pts, merged_cols = process_multiview_folder(root_dir)

    if merged_pts is None:
        print("[ERROR] No points merged. Exiting.")
        exit(0)

    print(f"[INFO] Merged cloud shape: {merged_pts.shape}")

    # 2) Create or load a MinkowskiEngine model for 3D features
    #model_3d = ExampleMinkUNet(in_nchannel=3, out_nchannel=32, D=3).to(device)
    # You could load pretrained weights if you have them:
    # model_3d.load_state_dict(torch.load("my_minkunet_weights.pth"))

    # 3) Extract features
    #features = extract_3d_features_minkowski(merged_pts, merged_cols, model_3d, voxel_size=0.01)
    print("[INFO] Extracted 3D feature vector:", features.shape)
    print(features)

