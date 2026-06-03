import torch
import numpy as np
from PIL import Image
import requests
import cv2
import easyocr
import matplotlib.pyplot as plt
import time
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")






# prepare image for the model

def increase_resolution(image_cv2):
    image = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    inputs = processor(image, return_tensors="pt")

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    # Image.fromarray(output).save("swin_output.png")
    return output

def visualize_easyocr(image_cv2, confidence_threshold=0.3):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # You can add other languages if needed

    # Read image
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    # Perform OCR
    results = reader.readtext(image_rgb)

    # Filter out text with low confidence
    filtered_results = [res for res in results if res[2] >= confidence_threshold]

    # Sort text top-to-bottom (by Y coordinate) and left-to-right within the same line
    sorted_results = sorted(filtered_results, key=lambda x: (x[0][0][1], x[0][0][0]))

    extracted_texts = []  # Store ordered text

    # Draw bounding boxes and extract text
    for (bbox, text, confidence) in sorted_results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        extracted_texts.append(text)  # Store text in list

        # Draw rectangle
        cv2.rectangle(image_rgb, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image_rgb, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the processed image
    # output_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output_easyocr.jpg', image_rgb)
    
    return extracted_texts

def get_object_text(image_cv2):
    try:
        start = time.time()
        image_cv2 = increase_resolution(image_cv2)
        # image_cv2 = increase_resolution(image_cv2)
        extracted_texts = visualize_easyocr(image_cv2)
        print("[INFO] OCR took {:.2f} seconds".format(time.time() - start))
        print("[INFO] Extracted texts: ", extracted_texts)
    except Exception as e:
        print("[ERROR] OCR failed due to", e)
        extracted_texts = []
    return extracted_texts

# url = "/home/rueiche/worldscribe/memory_data/2025-02-24_16-51-47/rgb_image.png"
# image = cv2.imread(url)
# print(get_object_text(image))
