import os
import glob
import json
import cv2
import torch
import numpy as np
import time
from PIL import Image

# from your local files:
from masking.GroundSamWraper import GroundSam_Wraper  # or wherever your GroundSam_Wraper is defined
from IPython import embed

# Example usage
DINO_CONFIG_FILE = "./masking/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT = "./masking/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "./masking/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

ground_sam = GroundSam_Wraper(
    config_file=DINO_CONFIG_FILE,
    grounded_checkpoint=DINO_CHECKPOINT,
    sam_checkpoint=SAM_CHECKPOINT,
    device="cuda"  # or "cpu"
)
    
    
def load_rgb_image(rgb_path):
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def crop_image_with_box(image, box):
    """
    Crops an image using the given bounding box.
    
    :param image: The input image as a NumPy array (H, W, 3).
    :param box: A list or array of [x_min, y_min, x_max, y_max].
    :return: Cropped image.
    """
    # Extract coordinates
    x_min, y_min, x_max, y_max = map(int, box)

    # Ensure coordinates are within image bounds
    H, W = image.shape[:2]
    x_min, x_max = max(0, x_min), min(W, x_max)
    y_min, y_max = max(0, y_min), min(H, y_max)

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image

def get_mask_by_hand_prompt(rgb, ground_sam: GroundSam_Wraper, text_prompt="the center object held by hands"):
    start_time = time.time()
    folder = ""
    
    print("******* debugging1 ********")
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
        time1 = time.time()
        masks, boxes = ground_sam.sam_bbox_with_cv2_input(
            image=rgb,
            image_size=(W_img, H_img),
            boxes_filt=boxes_filt
        )
        time2 = time.time()
        boxes = boxes.cpu().numpy()
        print(f"[INFO] boxes {boxes}")
        print(f"[INFO] getting masks takes {time2 - time1} seconds")
    except Exception as e:
        print("Exception in getting masks", e) 
        return None
    print("******* debugging2 ********")

    
    cropped_rgba_list = []
    if masks is None or len(masks) == 0:
        print(f"No mask found in the image. Skipping.")
        return None
    else:
        for i in range(masks.shape[0]):  # Iterate through each detected object
            mask_i = masks[i]
            
            # Ensure the mask is 2D (remove extra dimension if needed)
            if mask_i.ndim == 3:
                mask_i = mask_i[0]  # shape [H, W]
            mask_i = mask_i.cpu().numpy().astype(bool)

            # Get bounding box from the detected mask
            y_coords, x_coords = np.where(mask_i)  # Get nonzero mask indices
            if len(y_coords) == 0 or len(x_coords) == 0:
                continue  # Skip if mask is empty

            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            # Ensure bounding box is within image limits
            y_min, y_max = max(0, y_min), min(H_img, y_max)
            x_min, x_max = max(0, x_min), min(W_img, x_max)

            # Crop the region from the original image
            cropped_rgb = rgb[y_min:y_max, x_min:x_max].copy()

            # Create an empty alpha channel (same size as the cropped image)
            alpha_channel = np.zeros((cropped_rgb.shape[0], cropped_rgb.shape[1]), dtype=np.uint8)

            # Apply the mask within the cropped region
            mask_region = mask_i[y_min:y_max, x_min:x_max]
            alpha_channel[mask_region] = 255  # Set foreground pixels to fully opaque

            # Convert the cropped RGB image to RGBA (add alpha channel)
            cropped_rgba = np.dstack((cropped_rgb, alpha_channel))  # Shape (H, W, 4)
            cropped_rgba_list.append(cropped_rgba)
            # Save the cropped object as a transparent PNG
    
    for i, cropped_rgba in enumerate(cropped_rgba_list):
        cropped_path = os.path.join(folder, f"cropped_object_{i}.png")
        print(f"saving images to {cropped_path}")
        cv2.imwrite(cropped_path, cropped_rgba)
        print(f"[INFO] Saved cropped object with transparency to: {cropped_path}")

    # Combine multiple masks with OR
    # combined_mask = np.zeros((H_img, W_img), dtype=bool)
    # for i in range(masks.shape[0]):
    #     mask_i = masks[i]
    #     # If mask_i is [1, H, W], remove the extra dim
    #     if mask_i.ndim == 3:
    #         mask_i = mask_i[0]  # shape [H, W]
    #     mask_i = mask_i.cpu().numpy().astype(bool)
    #     combined_mask = np.logical_or(combined_mask, mask_i)

    # # Example: save the binary mask as a PNG (0=background, 255=foreground)
    # mask_save_path = os.path.join(folder, f"mask_{text_prompt.replace(' ', '_')}.png")
    # mask_255 = (combined_mask * 255).astype(np.uint8)
    # cv2.imwrite(mask_save_path, mask_255)
    # print(f"[INFO] Saved mask to: {mask_save_path}")


    # # This creates a color overlay on the original image
    # overlay = rgb.copy()
    # overlay[combined_mask] = [255, 0, 0]  # e.g. highlight in red
    # overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    # overlay_save_path = os.path.join(folder, f"overlay_{text_prompt.replace(' ', '_')}.png")
    # cv2.imwrite(overlay_save_path, overlay_bgr)
    
    # print(f"[INFO] Saved overlay to: {overlay_save_path}")

    end_time = time.time()
    print(f"[INFO] This process takes {end_time-start_time} seconds\n")
    return
    

if __name__ == "__main__":
    root_dir = "/home/rueiche/worldscribe/memory_data/2025-02-09_20-55-40"
    text_prompt = "the center object held by hands"

    # bgr_image_cv2 = cv2.imread("/home/rueiche/worldscribe/memory_data/2025-02-10_11-15-43/1/rgb_image.png")
    # rgb = cv2.cvtColor(bgr_image_cv2, cv2.COLOR_BGR2RGB)
    rgb_path = "/home/rueiche/worldscribe/memory_data/2025-02-10_11-15-43/1/rgb_image.png"
    rgb = load_rgb_image(rgb_path)
    get_mask_by_hand_prompt(rgb, ground_sam, text_prompt="the center object held by hands")
    
    # subfolders = sorted(glob.glob(os.path.join(root_dir, "*")))
    # for folder in subfolders:
    #     rgb_path = os.path.join(folder, "rgb_image.png")
    #     rgb = load_rgb_image(rgb_path)
    #     get_mask_by_hand_prompt(rgb, ground_sam, text_prompt)
        
    # process_multiview_folder(root_dir, ground_sam, text_prompt=text_prompt)