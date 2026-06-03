import json
import numpy as np
import open3d as o3d
import cv2

def save_colorized_point_cloud(color_bgr, depth_img, camera_data, output_ply="output_colored_pointcloud.ply"):
    # Example JSON camera parameters
    camera_params = {
        "origin": camera_data['camera_pose']['origin'],
        "x": camera_data['camera_pose']['x'],
        "y": camera_data['camera_pose']['y'],
        "z": camera_data['camera_pose']['z']
    }

    # ----------------------------------------------------------------------
    # 2) Parse camera JSON
    # ----------------------------------------------------------------------
    # camera_params = json.loads(camera_json)
    origin = np.array(camera_params["origin"], dtype=np.float32)
    Rx = np.array(camera_params["x"], dtype=np.float32)
    Ry = np.array(camera_params["y"], dtype=np.float32)
    Rz = np.array(camera_params["z"], dtype=np.float32)
    
    fx = camera_data['fx']
    fy = camera_data['fy']
    cx = camera_data['cx']
    cy = camera_data['cy']
    # We'll override the "max_depth" from the JSON and set our own truncation
    # below, since your real min depth is ~21 m, far above 5.0.

    # ----------------------------------------------------------------------
    # 3) Load images
    # ----------------------------------------------------------------------
    
    if color_bgr is None:
        print("ERROR: Could not load color image")
        return
    if depth_img is None:
        print("ERROR: Could not load depth image")
        return

    # Convert BGR -> RGB
    color_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # print("Color image shape:", color_img.shape, "dtype:", color_img.dtype)
    # print("Depth image shape:", depth_img.shape, "dtype:", depth_img.dtype)
    # print("Depth min:", depth_img.min(), "Depth max:", depth_img.max())

    # ----------------------------------------------------------------------
    # 4) Construct intrinsics from FOV
    # ----------------------------------------------------------------------
    height, width, _ = color_img.shape

    # print(f"Computed intrinsics -> fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # ----------------------------------------------------------------------
    # 5) Extrinsic (camera-to-world)
    # ----------------------------------------------------------------------
    R = np.column_stack([Rx, Ry, Rz])
    t = origin.reshape((3, 1))
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t.squeeze()

    # ----------------------------------------------------------------------
    # 6) Convert to Open3D Image
    # ----------------------------------------------------------------------
    o3d_color = o3d.geometry.Image(color_img)
    o3d_depth = o3d.geometry.Image(depth_img)

    # ----------------------------------------------------------------------
    # 7) Create RGBD with correct scale/trunc
    #    Since min depth is ~21019 (mm = ~21 m),
    #    we set scale=1000 to convert mm->meters,
    #    and set a larger truncation than 5.
    # ----------------------------------------------------------------------
    depth_scale = 1000.0
    depth_trunc = 50.0  # Something > 21 m

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    # ----------------------------------------------------------------------
    # 8) Reconstruct point cloud in camera coords
    # ----------------------------------------------------------------------
    pcd_camera = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )
    # print("Generated point cloud (camera frame):", pcd_camera)

    # ----------------------------------------------------------------------
    # 9) Transform to world coords if desired
    # ----------------------------------------------------------------------
    pcd_world = pcd_camera.transform(extrinsic)  # camera->world
    # print("Point cloud in world coords:", pcd_world)

    # ----------------------------------------------------------------------
    # 10) Save point cloud
    # ----------------------------------------------------------------------
    num_points = np.asarray(pcd_world.points).shape[0]
    if num_points == 0:
        print("[WARNING] Point cloud has 0 points. Check depth data / scaling.")
    o3d.io.write_point_cloud(output_ply, pcd_world)
    print(f"===Saved colored point cloud to: {output_ply}")

# if __name__ == "__main__":
#     color_bgr = cv2.imread("color.png", cv2.IMREAD_COLOR)
#     depth_img = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
#     reconstruct(color_bgr, depth_img)
