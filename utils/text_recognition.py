import cv2
import time
import numpy as np
import mediapipe as mp
import easyocr

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

def detect_text_with_boxes(image, finger_tip_location, confidence_threshold=0.4, distance_threshold=100, debug=True):
    """
    Detects text in an image, finds text closest to the fingertip, and visualizes the results.
    """
    reader = easyocr.Reader(['en'])  # Initialize OCR reader
    text_results = reader.readtext(image)
    
    finger_x, finger_y = finger_tip_location
    touched_text = None
    min_distance = float("inf")
    final_bbox = None
    
    for bbox, text, prob in text_results:
        if prob < confidence_threshold:
            continue  # Ignore low-confidence text
        
        # Extract bounding box points
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]

        # Compute bounding box center
        text_center_x = int(sum(x_coords) / 4)
        text_center_y = int(sum(y_coords) / 4)

        # Calculate Euclidean distance from fingertip
        distance = np.sqrt((finger_x - text_center_x) ** 2 + (finger_y - text_center_y) ** 2)

        # Skip texts that are too far from the fingertip
        if distance > distance_threshold:
            continue

        # Check if fingertip is inside bounding box
        if (min(x_coords) <= finger_x <= max(x_coords)) and (min(y_coords) <= finger_y <= max(y_coords)):
            # touched_text = text
            # final_bbox = bbox
            continue

        # Otherwise, find the closest text
        if distance < min_distance:
            min_distance = distance
            touched_text = text
            final_bbox = bbox

    # Visualization
    if debug:
        for bbox, text, prob in text_results:
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]

            # Draw bounding box
            color = (0, 0, 255) if text == touched_text else (0, 255, 0)
            cv2.polylines(image, [np.array(bbox, np.int32)], isClosed=True, color=color, thickness=2)

            # Display text and confidence
            cv2.putText(image, f"{text} ({prob:.2f})", (int(x_coords[0]), int(y_coords[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Mark fingertip location
        cv2.circle(image, finger_tip_location, radius=8, color=(255, 0, 0), thickness=-1)
        cv2.circle(image, finger_tip_location, radius=distance_threshold, color=(255, 0, 0), thickness=5)

        # Highlight final selected text
        cv2.putText(image, f"Selected: {touched_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite("text_debug123.png", image)
    return touched_text


# frame = cv2.imread("/Users/rueichechang/Projects/worldscribe/frame.png")
# rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # Adjust contrast and brightness
# alpha = 1.0  # Contrast control (1.0 means no change)
# beta = 50    # Brightness control (positive values increase brightness)
# rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=alpha, beta=beta)
# results = hands.process(rgb_frame)

# if results.multi_hand_landmarks and results.multi_handedness:
#     for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#         handedness = results.multi_handedness[idx].classification[0].label  # "Right" or "Left"
#         print("handedness", handedness)
#         if handedness == "Left":  # Process only right hand
#             h, w, _ = rgb_frame.shape
#             index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

#             print(f"Right-hand finger detected at: {x}, {y}")

#             # Draw fingertip point
#             cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

#             # Get the text the finger is pointing to
#             touched_text = detect_text_with_boxes(rgb_frame, (x, y))

#             if touched_text:
#                 cv2.putText(frame, touched_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                             0.8, (255, 0, 0), 2)

#     cv2.imwrite("text_output.png", frame)
# else:
#     print("Finger not found")
