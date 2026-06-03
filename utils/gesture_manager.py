import os
# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_MLIR"] = "1"

from datetime import datetime
import threading
import cv2
import mediapipe as mp
import time
import copy
import itertools
import numpy as np
from collections import deque, Counter
from utils.worldscribe_utils import is_point_inside_bbox
from utils.hand_gesture_recognition.model import KeyPointClassifier
from utils.hand_gesture_recognition.model import PointHistoryClassifier
from utils.audio_player import play_audio

# from hand_gesture_recognition.model import KeyPointClassifier
# from hand_gesture_recognition.model import PointHistoryClassifier
# from audio_player import play_audio


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def adjust_gamma(image, gamma=1.2):
    """ Adjust brightness using Gamma Correction """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def reduce_noise(image):
    """ Apply Gaussian Blur and Bilateral Filter to reduce noise """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Smooth noise
    bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)  # Preserve edges
    return bilateral

def adaptive_threshold(image):
    """ Convert to grayscale and apply adaptive thresholding """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def apply_clahe(img):
    """ Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_saturation(image, scale=1.0):
    """
    Adjusts the saturation of an image.
    
    Parameters:
    - image: Input BGR image.
    - scale: Saturation scale factor. (>1.0 increases, <1.0 decreases, 1.0 keeps it the same)
    
    Returns:
    - Adjusted BGR image.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Scale the Saturation channel
    h, s, v = cv2.split(hsv)
    s = np.clip(s * scale, 0, 255).astype(np.uint8)
    
    # Merge channels and convert back to BGR
    hsv_adjusted = cv2.merge([h, s, v])
    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    return adjusted_image

def calc_bounding_rect(image, landmarks, V=20):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    # Expand bounding box by V, but keep it within image bounds
    x_new = max(0, x - V)
    y_new = max(0, y - V)
    x_max = min(image_width, x + w + V)
    y_max = min(image_height, y + h + V)

    return [x_new, y_new, x_max, y_max]


class HandManager:
    def __init__(self, firebaseWriteManager, num_frame_check=24):

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier(model_path='index_point_history_classifier.tflite')
        self.thumb_point_history_classifier = PointHistoryClassifier(model_path='thumb_point_history_classifier.tflite')

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # send things to firebase for communicating with remote server
        self.firebaseWriteManager = firebaseWriteManager
        
        # parameters to determine what to collect
        self.num_frame_check = num_frame_check
        self.hands_history = deque(maxlen=self.num_frame_check)
        self.hand_gesture_history = deque(maxlen=self.num_frame_check)
        
        self.history_length = 8
        self.left_point_history = deque(maxlen=self.history_length) 
        self.right_point_history = deque(maxlen=self.history_length) 
        
        
        
        self.left_thumb_point_history = deque(maxlen=self.history_length) 
        self.right_thumb_point_history = deque(maxlen=self.history_length) 

        self.left_thumb_point_lock = False
        self.right_thumb_point_lock = False

        self.left_thumb_movement_history = deque(maxlen=int(self.history_length))
        self.right_thumb_movement_history = deque(maxlen=int(self.history_length))

        self.left_index_movement_history = deque(maxlen=int(self.history_length))
        self.right_index_movement_history = deque(maxlen=int(self.history_length))
        

        self.prev_hands_interacting = False
        self.hands_interacting = False
        self.send_a_request_for_hand_object = False
        self.storing_address = None

        self.prev_left_stable_index_movement = None
        self.curr_left_stable_index_movement = None
        self.prev_right_stable_index_movement = None
        self.curr_right_stable_index_movement = None

        self.stable_left_index_activities = deque(maxlen=int(self.history_length*2))
        self.stable_right_index_activities = deque(maxlen=int(self.history_length*2))


        self.prev_left_stable_activity = None
        self.curr_left_stable_activity = None
        self.prev_right_stable_activity = None
        self.curr_right_stable_activity = None

        self.left_change_time = time.time()
        self.right_change_time = time.time()

        self.prev_left_stable_thumb_movement = None
        self.curr_left_stable_thumb_movement = None
        self.prev_right_stable_thumb_movement = None
        self.curr_right_stable_thumb_movement = None

        # self.prev_left_object_swipe = None
        # self.prev_right_object_swipe = None
        self.curr_left_object_swipe = None
        self.curr_right_object_swipe = None

    def check_two_finger_swipes(self, hand_side_w_object):
        if hand_side_w_object == 'left':
            swipe_on_left_object  = False
            right_thumb_point_history = list(self.right_thumb_point_history)[-4:]
            right_point_history       = list(self.right_point_history)[-4:]

            thumb_filter = [coord[0] != 0 and coord[1] !=0 for coord in right_thumb_point_history]
            index_filter = [coord[0] != 0 and coord[1] !=0 for coord in right_point_history]
            
            if all(thumb_filter) and all(index_filter):
                swipe_on_left_object = True
            else:            
                swipe_on_left_object = False
            changed = self.curr_left_object_swipe != swipe_on_left_object
            self.curr_left_object_swipe = swipe_on_left_object
            return changed, swipe_on_left_object
        
        elif hand_side_w_object == 'right':
            swipe_on_right_object = False
            left_thumb_point_history = list(self.left_thumb_point_history)[-4:]
            left_point_history       = list(self.left_point_history)[-4:]
            
            thumb_filter = [coord[0] != 0 and coord[1] !=0 for coord in left_thumb_point_history]
            index_filter = [coord[0] != 0 and coord[1] !=0 for coord in left_point_history]
            
            if all(thumb_filter) and all(index_filter):
                swipe_on_right_object = True
            else:            
                swipe_on_right_object = False
            changed = self.curr_right_object_swipe != swipe_on_right_object
            self.curr_right_object_swipe = swipe_on_right_object
            return changed, swipe_on_right_object
    
    def classify_thumb_movement(self, frame):
        left_pre_processed_point_history_list = pre_process_point_history(
                frame, self.left_thumb_point_history)
        right_pre_processed_point_history_list= pre_process_point_history(
                frame, self.right_thumb_point_history)
        
        left_thumb_gesture_id = -1
        left_point_history_len = len(left_pre_processed_point_history_list)
        if left_point_history_len == (self.history_length * 2):
                left_thumb_gesture_id = self.thumb_point_history_classifier(
                    left_pre_processed_point_history_list)
        
        right_thumb_gesture_id = -1
        right_point_history_len = len(right_pre_processed_point_history_list)
        if right_point_history_len == (self.history_length * 2):
                right_thumb_gesture_id = self.thumb_point_history_classifier(
                    right_pre_processed_point_history_list)

        self.left_thumb_movement_history.append(left_thumb_gesture_id)
        self.right_thumb_movement_history.append(right_thumb_gesture_id)


        left_count = Counter(list(self.left_thumb_movement_history)).most_common()
        right_count = Counter(list(self.right_thumb_movement_history)).most_common()

        thumb_point_history_classifier_labels = ['stop', 'move', 'move', 'None']

        left_thumb_movement = thumb_point_history_classifier_labels[left_count[0][0]]
        right_thumb_movement = thumb_point_history_classifier_labels[right_count[0][0]]

        # left_movement = thumb_point_history_classifier_labels[left_thumb_gesture_id]
        # right_movement = thumb_point_history_classifier_labels[right_thumb_gesture_id]

        return left_thumb_movement, right_thumb_movement
    
    def check_thumb_movement_state_change(self, left_movement, right_movement):
    
        left_thumb_change = False
        right_thumb_change = False
        time_interval = 5

        def unlock_thumb(hand_side):
            if hand_side == "left": self.left_thumb_point_lock = False
            else: self.right_thumb_point_lock = False
        # if self.curr_left_stable_thumb_movement != left_movement:
        #     left_thumb_change = True
        #     self.curr_left_stable_thumb_movement = left_movement
        # if self.curr_right_stable_thumb_movement!= right_movement:
        #     right_thumb_change = True
        #     self.curr_right_stable_thumb_movement = right_movement
            
        if self.curr_left_stable_thumb_movement != left_movement:
            if self.curr_left_stable_thumb_movement == 'stop' and left_movement == 'move':
                # time.time() - self.left_change_time < time_interval:
                self.curr_left_stable_thumb_movement = left_movement
                self.left_thumb_point_lock = True
                threading.Timer(2.0, unlock_thumb, args=('left',)).start()
                left_thumb_change = True
            else:
                self.prev_left_stable_thumb_movement = self.curr_left_stable_thumb_movement
                self.curr_left_stable_thumb_movement = left_movement
                left_thumb_change = False


        # print(f'right_movement   {self.right_thumb_point_lock}\n'*200)
        if self.curr_right_stable_thumb_movement != right_movement:
            # print(f'right_movement   {right_movement}\n'*200)
            if self.curr_right_stable_thumb_movement == 'stop' and right_movement == 'move':
                # time.time() - self.left_change_time < time_interval:
                self.curr_right_stable_thumb_movement = right_movement
                self.right_thumb_point_lock = True
                threading.Timer(2.0, unlock_thumb, args=('right',)).start()
                right_thumb_change = True
            else:
                self.prev_right_stable_thumb_movement = self.curr_right_stable_thumb_movement
                self.curr_right_stable_thumb_movement = right_movement
                right_thumb_change = False

        # print(f"[THUMB INFO] left hand: {left_movement}")
        # print(f"[THUMB INFO] right hand: {right_movement}")

        # self.curr_left_stable_thumb_movement = left_movement
        # self.curr_right_stable_thumb_movement = right_movement
        
        return left_thumb_change, right_thumb_change

    def is_index_movement_stable(self, hand_side):
        d = self.stable_left_index_activities if hand_side == 'left' else self.stable_right_index_activities
        if len(d) == int(self.history_length*4):
            if len(set(d)) == 1 and d[-1] is not None:
                return True
            return False
        else:
            return False

    def classify_index_movement(self, frame):
        left_pre_processed_point_history_list = pre_process_point_history(
                frame, self.left_point_history)
        right_pre_processed_point_history_list= pre_process_point_history(
                frame, self.right_point_history)
        
        left_finger_gesture_id = -1
        left_point_history_len = len(left_pre_processed_point_history_list)
        if left_point_history_len == (self.history_length * 2):
                left_finger_gesture_id = self.point_history_classifier(
                    left_pre_processed_point_history_list)
        
        right_finger_gesture_id = -1
        right_point_history_len = len(right_pre_processed_point_history_list)
        if right_point_history_len == (self.history_length * 2):
                right_finger_gesture_id = self.point_history_classifier(
                    right_pre_processed_point_history_list)

        self.left_index_movement_history.append(left_finger_gesture_id)
        self.right_index_movement_history.append(right_finger_gesture_id)

        point_history_classifier_labels = ['stop', 'move', 'None']

        left_index_movement_history_list = list(self.left_index_movement_history)[-4:]
        right_index_movement_history_list = list(self.right_index_movement_history)[-4:]


        left_most_common_fg_id = Counter(left_index_movement_history_list).most_common()
        right_most_common_fg_id = Counter(right_index_movement_history_list).most_common()

        left_index_movement = point_history_classifier_labels[left_most_common_fg_id[0][0]]
        right_index_movement = point_history_classifier_labels[right_most_common_fg_id[0][0]]

        # left_movement = point_history_classifier_labels[left_finger_gesture_id]
        # right_movement = point_history_classifier_labels[right_finger_gesture_id]

        return left_index_movement, right_index_movement



    def check_index_movement_state_change(self, left_index_movement, right_index_movement):
        
        left_index_change = False
        right_index_change = False

        if self.curr_left_stable_index_movement != left_index_movement:
            self.prev_left_stable_index_movement = self.curr_left_stable_index_movement
            self.curr_left_stable_index_movement = left_index_movement
            left_index_change = True
            
        if self.curr_right_stable_index_movement!= right_index_movement:
            self.prev_right_stable_index_movement = self.curr_right_stable_index_movement
            self.curr_right_stable_index_movement = right_index_movement
            right_index_change = True
            
        # print(f"[INDEX INFO] left hand: {left_index_movement}")
        # print(f"[index INFO] right hand: {right_index_movement}")
        
        return left_index_change, right_index_change
    
    def check_over_n_with_length_two(self, lst, n):
        many_hands_count   = sum(1 for sublist in lst if len(sublist) > 2)
        both_hands_count   = sum(1 for sublist in lst if len(sublist) == 2)
        single_hands_count = sum(1 for sublist in lst if len(sublist) == 1)
        zero_hands_count   = sum(1 for sublist in lst if len(sublist) == 0)
        
        if both_hands_count >= n: 
            return True
        elif zero_hands_count >= n: 
            return False 
        else: 
            return self.hands_interacting

    def recognize_each_hand_stable_gestures(self):
        hand_gesture_history = list(self.hand_gesture_history)
        if len(hand_gesture_history) < self.num_frame_check: 
            # print(f"[DEBUG] {hand_gesture_history}")
            return None, None

        left_hands_activities = []
        right_hands_activities = []

        for both_hands_activities in hand_gesture_history:
            if not both_hands_activities:  # No hand detected
                left_hands_activities.append(None)
                right_hands_activities.append(None)
                continue

            left_gesture = None
            right_gesture = None

            for data in both_hands_activities:
                which_hand = data["which_hand"]
                gesture = data["gesture"]
                
                if which_hand == "left":
                    left_gesture = gesture
                elif which_hand == "right":
                    right_gesture = gesture
            
            # Ensure both hands' lists are of equal length
            left_hands_activities.append(left_gesture)
            right_hands_activities.append(right_gesture)
        
        left_activity = self.get_most_common_activities('left', left_hands_activities)
        right_activity = self.get_most_common_activities('right', right_hands_activities)

        return left_activity, right_activity

    def get_most_common_activities(self, hand_side, activities, n=4):
        if not activities:
            return None

        full_activities = list(activities)[-18:]
        mid_activities = full_activities[-12:]  # Last 16 frames
        few_activities = full_activities[-6:]   # Last 8 frames
        mini_activities = full_activities[-3:]  # Last 4 frames

        full_counts = Counter(full_activities)
        mid_counts = Counter(mid_activities)
        few_counts = Counter(few_activities)
        mini_counts = Counter(mini_activities)

        current_stable_activity = (
            self.curr_left_stable_activity if hand_side == 'left' 
            else self.curr_right_stable_activity
        )

        # If all activities are None, return None
        if set(mid_activities) == {None}:
            # print("All activities are None")
            return None

        # If no stable activity, assign based on mid_counts
        if current_stable_activity is None:
            for gesture in ['grab', 'pointer', 'touch_explore']:
                if set(mid_counts.keys()).issubset({gesture, None}) and mid_counts[gesture] > n:
                    return gesture
            return few_counts.most_common(1)[0][0] if few_counts and few_counts.most_common(1)[0][1] > n else None

        # --- Special Handling for Pointer ---
        if current_stable_activity == "pointer":
            if "pointer" not in mini_activities:
                # If "pointer" is not detected in the last 4 frames, switch quickly
                next_activity = mini_counts.most_common(1)[0][0] if mini_counts else None
                print(f"Fast transition from pointer to {next_activity}")
                return next_activity
            return "pointer"  # Otherwise, remain in pointer mode

        # --- Stricter Condition for Entering Pointer ---
        # if "pointer" in few_activities and mid_counts["pointer"] >= 12:
        if "pointer" in few_activities and full_counts["pointer"] >= 16:
            # Only enter pointer if it has been seen in at least 12 out of 16 frames
            return "pointer" 

        # --- Standard Handling for Other Gestures ---
        if set(mid_counts.keys()).issubset({current_stable_activity, None}):
            return current_stable_activity

        if current_stable_activity in few_activities:
            return current_stable_activity

        next_activity = few_counts.most_common(1)[0][0] if few_counts else None
        print(f"Transitioning from {current_stable_activity} to {next_activity}")
        return next_activity


    def check_hand_event_state_change(self, left_activity, right_activity):
        
        left_hand_change = False
        right_hand_change = False

        if self.curr_left_stable_activity != left_activity:
            self.prev_left_stable_activity = self.curr_left_stable_activity
            self.curr_left_stable_activity = left_activity
            # if left_activity: 
            left_hand_change = True

        if self.curr_right_stable_activity != right_activity:
            self.prev_right_stable_activity = self.curr_right_stable_activity
            self.curr_right_stable_activity = right_activity
            # if right_activity: 
            right_hand_change = True

        # print(f"[GESTURE INFO] left hand: {left_activity}")
        # print(f"[GESTURE INFO] right hand: {right_activity}")
        
        if left_hand_change: self.left_change_time = time.time()
        if right_hand_change: self.right_change_time = time.time()
        
        return left_hand_change, right_hand_change

    def check_interact_with_something(self, frame_info):
        # [[[123],[456]], [[123],[456]], [[123],[456]]]
        hands_history = list(self.hands_history)
        frame_id = frame_info['frame_id']
        if frame_id % self.num_frame_check != 0: return False
        if len(hands_history) < self.num_frame_check: return False

        # self.hands_interacting = all(len(sublist) == 2 for sublist in hands_history)
        self.hands_interacting = self.check_over_n_with_length_two(hands_history, len(hands_history)-1)
        
        # detect state change
        hands_left = False
        if self.prev_hands_interacting != self.hands_interacting:
            self.prev_hands_interacting = self.hands_interacting
            if self.hands_interacting:
                self.storing_address = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                # self.send_a_request_for_hand_object = True
                # play_audio("hands_detected")
                
            else:
                # play_audio("hands_left")
                hands_left = True
                # self.send_a_request_for_hand_object = True

        
        return hands_left
    
    def storing_image_in_remote(self, frame_info):
        if self.hands_interacting and len(self.hands_history[-1])>0:
            packet = {
                        # 'rgb_frame_string'      : frame_info['camera_data']['rgb_frame_string'],
                        # 'depth_frame_string'    : frame_info['camera_data']['depth_frame_string'],
                        'uuid'                  : frame_info['uuid'],
                        'ids'                   : frame_info['ids'],
                        'object_classes'        : frame_info['object_classes'],
                        'frame_id'              : frame_info['frame_id'],
                        'timestamp'             : self.storing_address,
                        'timestamp_s'           : time.time(),
                        'current_caption'       : frame_info['current_caption'],
                        'current_caption_source': frame_info['current_caption_source'],
                        'camera_data'           : frame_info['camera_data'],
                    }        
            self.firebaseWriteManager.send_memory_image_to_server(packet)
            return True
        return False

    def hand_management(self, rgb_frame):
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        # Adjust contrast and brightness
        # alpha = 1.0  # Contrast control (1.0 means no change)
        # beta = 50    # Brightness control (positive values increase brightness)
        # rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=alpha, beta=beta)
        

        # rgb_frame = apply_clahe(rgb_frame)  # Enhance contrast
        # rgb_frame = adjust_gamma(rgb_frame, gamma=1.2)  # Brightness correction
        # rgb_frame = reduce_noise(rgb_frame)

        # rgb_frame = adjust_saturation(rgb_frame, 1.5)
        hand_data = []  # Store information about detected hands

        with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.1,
                model_complexity=1
                ) as hands:

            # Process the frame and detect hands
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                h, w, _ = rgb_frame.shape  # Get frame dimensions
                bottom_threshold = 4 * h // 5
                
                for hand_index, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, 
                                                                              results.multi_handedness)):

                    
                    # Get thumb tip and index finger tip landmarks
                    wrist_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    wrist_x, wrist_y = int(wrist_tip.x * w), int(wrist_tip.y * h)
                    # is_wrist_bottom = wrist_y >= bottom_threshold
                    if wrist_y >= bottom_threshold: 
                        continue

                    # brect = calc_bounding_rect(rgb_frame, hand_landmarks)

                    thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Convert normalized coordinates to pixel coordinates
                    thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)

                    # Determine if the hand is Left or Right
                    detected_hand = handedness.classification[0].label  # 'Left' or 'Right'
                    which_hand = "left" if detected_hand == "right" else "right"
                    # corrected_hand = "Right" if detected_hand == "Left" else "Left"

                    landmark_list = calc_landmark_list(rgb_frame, hand_landmarks)
                    h, w, _ = rgb_frame.shape
                    image_center_x = w // 2
                    wrist_x = landmark_list[0][0]
                    is_correct_side = (
                        (which_hand == "left" and wrist_x < image_center_x) or
                        (which_hand == "right" and wrist_x >= image_center_x)
                    )
                    


                    # Store data in a structured format
                    hand_info = {
                        "which_hand"     : which_hand,
                        "wrist_x"        : wrist_x,
                        "thumb_tip"      : (thumb_tip_x, thumb_tip_y),
                        "index_tip"      : (index_tip_x, index_tip_y),
                        "is_correct_side": is_correct_side
                    }

                    hand_data.append(hand_info)
                    # print(hand_info)

            if len(hand_data) == 2:
                leftmost, rightmost = sorted(hand_data, key=lambda h: h["wrist_x"])

                # Ensure the leftmost hand is labeled as "left" and rightmost as "right"
                leftmost["which_hand"], rightmost["which_hand"] = "left", "right"
            elif len(hand_data) == 1:
                if not hand_data[0]['is_correct_side']: 
                    hand_data[0]['which_hand'] = 'right' if hand_data[0]['which_hand'] == 'left' else 'left'

            elif len(hand_data) ==0:
                return None, rgb_frame
            
            self.hands_history.append(hand_data)
            
            
            return results, rgb_frame

        return None, rgb_frame


    def annotate_hands(self, frame, results, gesture_data):
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        color = (255, 0, 0)  # Green color for text

        # deal with left hand
        which_hand = 'left'
        text_position = (50, 100)         
        cv2.putText(frame, f"Stable: {self.curr_left_stable_activity}", text_position, font, font_scale, color, font_thickness)
        
        # if left_movement:
        text_position = (50, 200)
        cv2.putText(frame, f"Index: {self.curr_left_stable_index_movement}", text_position, font, font_scale, (0, 0, 255), font_thickness)

        # if left_thumb_movement:
        text_position = (50, 300)
        cv2.putText(frame, f"Swipe: {self.curr_left_object_swipe}", text_position, font, font_scale, (255, 255, 0), font_thickness)
            
        # deal with right hand
        which_hand = 'right'
        text_position = (frame.shape[1] - 300, 100)         
        cv2.putText(frame, f"Stable: {self.curr_right_stable_activity}", text_position, font, font_scale, color, font_thickness)
        
        # if right_movement:
        text_position = (frame.shape[1] - 300, 200)         
        cv2.putText(frame, f"Index: {self.curr_right_stable_index_movement}", text_position, font, font_scale, (0, 0, 255), font_thickness)

        # if right_thumb_movement:
        text_position = (frame.shape[1] - 300, 300)         
        cv2.putText(frame, f"Swipe: {self.curr_right_object_swipe}", text_position, font, font_scale, (255, 255, 0), font_thickness)
        
        
        text_position = (50, 800)
        color = (0, 0, 255)
        interaction_text = "True" if self.hands_interacting else "False"
        cv2.putText(frame, f"Bimanual: {interaction_text}", text_position, font, font_scale, color, font_thickness)


        if not results:
            return frame
        elif results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks
                
                landmark_color = self.mp_draw.DrawingSpec(color=(0, 0, 0), thickness=2)  # Green landmarks
                connection_color = self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)  # Blue connections


                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, landmark_color, connection_color)

                brect = calc_bounding_rect(frame, hand_landmarks)

                cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 3)
                
                # Get frame dimensions
                h, w, _ = frame.shape

                # Extract thumb tip (Landmark 4)
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Extract index finger tip (Landmark 8)
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)

                # Draw circle for thumb tip (Green)
                frame = cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (0, 255, 0), -1)

                # Draw circle for index finger tip (Yellow)
                frame = cv2.circle(frame, (index_tip_x, index_tip_y), 10, (0, 255, 255), -1)

                # Display the coordinates
                frame = cv2.putText(frame, f'Thumb ({thumb_tip_x}, {thumb_tip_y})',
                                    (thumb_tip_x + 10, thumb_tip_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                frame = cv2.putText(frame, f'Index ({index_tip_x}, {index_tip_y})',
                                    (index_tip_x + 10, index_tip_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # Define text properties
        if gesture_data:
            for item in gesture_data:
                gesture = item['gesture']
                which_hand = item['which_hand']

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                color = (0, 255, 0)  # Green color for text
                
                # Define text position based on which hand is detected
                text_position = (50, 50) if which_hand == "left" else (frame.shape[1] - 300, 50) 
                
                # Put text on frame
                cv2.putText(frame, f"{which_hand.capitalize()}: {gesture}", text_position, font, font_scale, color, font_thickness)
        


    
        return frame

    def hand_gesture_recognition(self, frame, results):
        gesture_classes = ['touch_explore', 
                           'touch_explore', 
                           'pointer', 
                           'pointer', 
                           'grab', 
                           'grab']
        # left_rest
        # right_rest
        # left_pointer
        # right_pointer
        # left_grab
        # right_grab

        if not results: 
            self.hand_gesture_history.append([])
            return []

        if not results.multi_hand_landmarks or not results.multi_handedness:
            self.hand_gesture_history.append([])
            return []

        # Step 1: Extract hand information (landmarks, handedness, and wrist position)
        gesture_data = []
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

            # Extract wrist x-coordinate (Landmark 0)
            wrist_x = landmark_list[0][0]

            # Assign initial handedness based on MediaPipe classification
            detected_hand = handedness.classification[0].label.lower()
            which_hand = "left" if detected_hand == "right" else "right"

            h, w, _ = frame.shape
            image_center_x = w // 2
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)
            
            
            is_correct_side = (
                (which_hand == "left" and wrist_x < image_center_x) or
                (which_hand == "right" and wrist_x >= image_center_x)
            )
            

            # Store hand data
            gesture_data.append({
                "gesture": gesture_classes[hand_sign_id],
                "which_hand": which_hand,
                "wrist_x": wrist_x,
                "index_tip_location": (index_tip_x, index_tip_y),
                "index_point": landmark_list[8],
                "thumb_point": landmark_list[4],
                "is_correct_side": is_correct_side,
                "hand_region": calc_bounding_rect(frame, hand_landmarks)
            })


        
        # Step 2: Correct misclassified handedness if two hands are detected
        if len(gesture_data) == 2:
            leftmost, rightmost = sorted(gesture_data, key=lambda h: h["wrist_x"])

            # Ensure the leftmost hand is labeled as "left" and rightmost as "right"
            leftmost["which_hand"], rightmost["which_hand"] = "left", "right"
        elif len(gesture_data) == 1:
            if not gesture_data[0]['is_correct_side']: 
                gesture_data[0]['which_hand'] = 'right' if gesture_data[0]['which_hand'] == 'left' else 'left'
        
        self.hand_gesture_history.append(gesture_data)

        # for data in gesture_data:
        #     if data['gesture'] == 'touch_explore' or data['gesture'] == 'grab':
        #         if data['which_hand'] == 'left':
        #             self.left_point_history.append(data['index_point'])
        #         elif data['which_hand'] == 'right':
        #             self.right_point_history.append(data['index_point'])
        #     else:
        #         if data['which_hand'] == 'left':
        #             self.left_point_history.append([0,0])
        #         elif data['which_hand'] == 'right':
        #             self.right_point_history.append([0,0])

            # if data['gesture'] in ['grab', 'touch_explore'] and is_point_inside_bbox(data['thumb_point'], data['hand_region']):
            #     if data['which_hand'] == 'left':
            #         if self.left_thumb_point_lock: self.left_thumb_point_history.append([0,0])
            #         else: self.left_thumb_point_history.append(data['thumb_point'])
            #     elif data['which_hand'] == 'right':
            #         if self.right_thumb_point_lock: self.right_thumb_point_history.append([0,0])
            #         else: self.right_thumb_point_history.append(data['thumb_point'])
            # else:
            #     if data['which_hand'] == 'left':
            #         self.left_thumb_point_history.append([0,0])
            #     elif data['which_hand'] == 'right':
            #         self.right_thumb_point_history.append([0,0])

        if len(gesture_data) == 2:
            left_data  = [data for data in gesture_data if data['which_hand'] == 'left'][0]
            right_data = [data for data in gesture_data if data['which_hand'] == 'right'][0]

            if right_data['gesture'] in ['grab', 'touch_explore'] and left_data['gesture'] in ['grab', 'touch_explore']:
                thumb_point = left_data['thumb_point'] if is_point_inside_bbox(left_data['thumb_point'], right_data['hand_region']) else [0,0]
                self.left_thumb_point_history.append(thumb_point)
                index_point = left_data['index_point'] if is_point_inside_bbox(left_data['index_point'], right_data['hand_region']) else [0,0]
                self.left_point_history.append(index_point)

                

            if left_data['gesture'] in ['grab', 'touch_explore'] and right_data['gesture'] in ['grab', 'touch_explore']:
                thumb_point = right_data['thumb_point'] if is_point_inside_bbox(right_data['thumb_point'], left_data['hand_region']) else [0,0]
                self.right_thumb_point_history.append(thumb_point)
                index_point = right_data['index_point'] if is_point_inside_bbox(right_data['index_point'], left_data['hand_region']) else [0,0]
                self.right_point_history.append(index_point)

            # print(f"self.left_thumb_point_history {self.left_thumb_point_history}\n \
            #       self.left_point_history {self.left_point_history}\n \
            #       self.right_thumb_point_history {self.right_thumb_point_history}\n \
            #       self.right_point_history {self.right_point_history} \n \
            #       "*100)



        # Step 3: Print gesture information
        # for hand in gesture_data:
        #     print(f"[GESTURE INFO] {hand['which_hand']} hand: {hand['gesture']}, wrist_x: {hand['wrist_x']}")

        return gesture_data
    
    def get_index_finger_tip(self, frame, gesture_data, hand_side="right"):
        for data in gesture_data:
            if data['which_hand'] == hand_side:
                return data['index_tip_location']
        return None
    
    def get_hand_region(self, frame, gesture_data, hand_side="right"):
        for data in gesture_data:
            if data['which_hand'] == hand_side:
                return data['hand_region']
        return None


if __name__ == '__main__':
    frame = cv2.imread("frame.jpg")
    handManager = HandManager(None)
    left_hands_activities = ['rest', 'rest', 'rest', 'grab', None, 'grab', 'grab', 'grab', 'grab', None]
    most_common = handManager.get_most_common_activities(left_hands_activities, n=2)
    print(most_common)  # Output: ['grab']
    # results, rgb_frame = handManager.hand_management(frame)
    # handManager.hand_gesture_recognition(results)
    # frame = handManager.annotate_hands(frame,results)
    # cv2.imwrite("frame_hand.jpg", frame)