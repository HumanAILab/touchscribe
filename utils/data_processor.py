"""dataprocessor"""
import threading
import time
import queue
import uuid
import copy
import supervision as sv
import numpy as np
from datetime import datetime
from collections import Counter, deque
import cv2

from utils.user_preference import SENTENCE_LENGTH_10, SENTENCE_LENGTH_20, SENTENCE_LENGTH_5, AGENT_NAME
from utils.user_preference import UserPreference
from utils.gesture_manager import HandManager
from utils.worldscribe_utils import encode_image_to_base64, get_caption_from_yolo_classes_adv, is_point_inside_bbox, synthesize_hand_caption
from utils.firebase.firebase_manager import FirebaseWriteManager, angle_difference
from utils.firebase.firestore_manager import FirestoreManager
from utils.nlp_process import concise_sentence, get_sentence_similarity_spacy
from utils.speech import TTSManager
from utils.text_recognition import detect_text_with_boxes
from utils.classes import COCO_CLASSES, OBJECT365_CLASSES
from ultralytics import YOLO
from similarity.VGG16_similarity_Xing import get_frame_similarity
from color_utils import get_pointing_color
SOUND_CONFIDENCE_THRES = 0.3 # sound confidence
SIMILARITY_THRESHOLD = 0.5 # similarity threshold
TURN_DEGREE_THRESHOLD = 30 # the descriptions stop if the user turns more than 30 degree
MESSY_ELEMENT = [99999999, 99999999, 99999999]




class DataProcessor:
    def __init__(self, use_mask=False):
        # ignore this, this is used to calculate the average time of YOLO, which was reported in the paper
        # self.yolo_num = 0
        # self.yolo_time = 0

        # logging each user's data everytime the program is run
        self.firestoreManager = FirestoreManager()
        # tracking live tts threads and stop them if needed
        self.live_tts_threads = []
        self.stop_threads_event = threading.Event()

        # This is YOLO model and annotators for visualization (e.g., bounding boxes), which will be determined later in process_data_by_yolo
        self.model = None
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.frame_id = 0
        self.current_frame = ""
        self.frame_history = {}
        self._caption_queue = "start"
        self._lock = threading.Lock()

        # track which sounds are happening
        self.sound_event_track = {}
        # track which sound manipulations are being applied
        self.ongoing_sound_manipulation = {}
        # This stores the user's preference on their smartphone
        self.user_preference = UserPreference(self)
        # self.user_preference.custom_classes = COCO_CLASSES

        # store the current frame_info
        self.frame_info = None

        # record time when the system starts
        self.start_time = 0
        # The current spoken descriptions
        self.current_caption = ""
        self.current_caption_source = ""

        self.interrupted_caption = ""
        self.interrupted_caption_source = ""

        # This manager writes and retreives data from the firebase
        self.firebaseWriteManager = FirebaseWriteManager()
        self.handManager = HandManager(self.firebaseWriteManager)

        # This manager supports text to speech and deal with the interruption
        self.ttsManager = TTSManager(self.firebaseWriteManager)

        self.last_time_speak = 0
        self.user_current_clock = 0
        self.user_current_degree = None
        self.user_prev_degree = None
        self.is_talking_less = False
        self.num_frame_to_check = 6

        self.force_to_read = False

        
        self.moondream_spoken_uuid = []
        self.gpt_spoken_uuid = []
        
        self.touched_text = {}


        self.new_object_check = True
        self.is_new_object_on_left = False
        self.is_new_object_on_right = False

        # The number of frames that does not have consistent object composition in the scene
        # self.idle_frame_num = 0
        self.prev_scene_state = {
            "state": None,
            "sources_used": [],
            "descriptions": [],
            "frame_id": 0,
            "uuid": None
        }
        self.curr_scene_state = self.prev_scene_state

        # Simulate short-term memory that do not overlap the previous descriptions
        self.caption_history = []

        # description can be stopped due to either the user's manipulation or they turn around over 30 degree
        self.stop_streaming_reason = ""

        self.ids_stable_state = [10000]
        self.long_ids_stable_state = [10000]
        self.prev_state_details = {
                "thread": None,
                "state" : None,
                "ids"   : self.ids_stable_state,
                "frame" : None,
                "frame_id" : self.frame_id, 
                "state_changed_time" : time.time()
            }

        self.object_text_buffer = []
        self.system_message = []
        self.text_system_message = []

        self.user_moving_history = deque(maxlen=16)
        self.user_is_moving = False

    # The generated descriptions will be put into the caption queue and retrieved later
    @property
    def caption_queue(self) -> str:
        with self._lock:
            return self._caption_queue

    @caption_queue.setter
    def caption_queue(self, value: str):
        with self._lock:
            self._caption_queue = value
    
    def check_user_moving(self):
        return any(self.user_moving_history)

    def initialize_frame_info(self, frame, server_data_dict):
        """
        Initializes frame information dictionary with metadata and placeholders.
        """
        return {
            "frame_id": self.frame_id,
            "frame": frame,
            "frame_shape": frame.shape,
            "boxes": [],
            "ids": [],
            "object_classes": [],
            "images": [],
            "masks": [],
            "confs": [],
            "yolo_caption": None,
            "text": None,
            "uuid": str(uuid.uuid4().hex),
            "camera_data": server_data_dict["camera_data"],
            "current_caption":None,
            "current_caption_source": None
        }
    
    def process_data_by_hands(self, frame, server_data_dict):
        if self.start_time == 0:
            self.start_time = time.time()

        self.frame_id += 1
        frame_info = self.initialize_frame_info(frame, server_data_dict)
        self.current_frame = frame
        self.user_moving_history.append(self.user_is_moving)
        self.user_is_moving = False
        
        
        # cv2.imwrite("frame.png", frame)
        annotated_frame = copy.deepcopy(frame)

        frame.flags.writeable = False
        hands_results, hand_frame = self.handManager.hand_management(frame)
        frame.flags.writeable = True

        if_left_hand_change  = False
        if_right_hand_change = False

        # print(f"[INFO] ", hands_results)
        # if hands_results:
        # self.handManager.storing_image_in_remote(frame_info=frame_info)
        # hands_left = self.handManager.check_interact_with_something(frame_info=frame_info)
        

        gesture_data = self.handManager.hand_gesture_recognition(annotated_frame, hands_results)
        
        left_stable_activity, right_stable_activity = self.handManager.recognize_each_hand_stable_gestures()
        if_left_hand_change, if_right_hand_change = self.handManager.check_hand_event_state_change(left_stable_activity, right_stable_activity)
        
        if left_stable_activity is None and right_stable_activity is None \
            or if_left_hand_change and left_stable_activity is None \
            or if_right_hand_change and right_stable_activity is None: 
            self.curr_scene_state["state"] = "turn"
            
            self.reset_all_text_buffer()
            if if_left_hand_change and left_stable_activity is None:
                self.system_message.append({'text': "your left hand just out",  
                                                    'event':'see_hand', 
                                                    'timestamp_s': time.time(), 
                                                    'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},)
                if right_stable_activity is None:
                    self.system_message.append({'text': "your right hand just out",  
                                                        'event':'see_hand', 
                                                        'timestamp_s': time.time(), 
                                                        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},)

            if if_right_hand_change and right_stable_activity is None: 
                self.system_message.append({'text': "your right hand just out",  
                                                    'event':'see_hand', 
                                                    'timestamp_s': time.time(), 
                                                    'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},)
                if left_stable_activity is None:
                    self.system_message.append({'text': "your left hand just out",  
                                                        'event':'see_hand', 
                                                        'timestamp_s': time.time(), 
                                                        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},)
            # print(f"self.system_message {[item['text'] for item in self.system_message]}\n" * 1000)
            
            text_list = [item['text'] for item in self.system_message]
            if 'your right hand just out' in text_list and 'your left hand just out' in text_list: 
                self.provide_reason_to_stop("turn")
                self.force_to_read = True
            
        # check index movement
        left_index_movement = None 
        right_index_movement = None
        left_index_change = None
        right_index_change = None
        
        if left_stable_activity == 'touch_explore' or right_stable_activity == 'touch_explore' or \
            left_stable_activity == 'grab' or right_stable_activity == 'grab':
            left_index_movement, right_index_movement = self.handManager.classify_index_movement(frame)
            left_index_change, right_index_change = self.handManager.check_index_movement_state_change(left_index_movement, right_index_movement)
            # print("[INDEX] left: ", left_index_movement)
            # print("[INDEX] right: ", right_index_movement)

            if left_index_change:
                self.handManager.stable_left_index_activities.clear()
            elif left_stable_activity in ['touch_explore', 'grab']:
                self.handManager.stable_left_index_activities.append(left_index_movement)

            if right_index_change:
                self.handManager.stable_right_index_activities.clear()
            elif right_stable_activity in ['touch_explore', 'grab']:
                self.handManager.stable_right_index_activities.append(left_index_movement)
    


        # check thumb movement
        left_thumb_movement = None 
        right_thumb_movement = None
        left_thumb_change = None
        right_thumb_change = None
        # print(f"[left_stable_activity] {left_stable_activity}")
        # print(f"[right_stable_activity] {right_stable_activity}")
        if left_stable_activity in ['grab'] and right_stable_activity in ['grab', 'touch_explore']:
            changed, swipe_on_left_object =  self.handManager.check_two_finger_swipes('left')
            if changed and swipe_on_left_object:
                self.firebaseWriteManager.send_thumb_gesture({  "timestamp" : datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                "timestamp_s": time.time(),
                                                                "left_object_change": swipe_on_left_object, 
                                                                "right_object_change": False,
                                                            })
                self.provide_reason_to_stop('text_interrupt')
        elif left_stable_activity in ['grab', 'touch_explore'] and right_stable_activity in ['grab']:
            changed, swipe_on_right_object = self.handManager.check_two_finger_swipes('right')
            if changed and swipe_on_right_object:
                self.firebaseWriteManager.send_thumb_gesture({  "timestamp" : datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                "timestamp_s": time.time(),
                                                                "left_object_change": False, 
                                                                "right_object_change": swipe_on_right_object,
                                                            })
                self.provide_reason_to_stop('text_interrupt')
        #     left_thumb_movement, right_thumb_movement = self.handManager.classify_thumb_movement(frame)
        #     left_thumb_change, right_thumb_change = self.handManager.check_thumb_movement_state_change(left_thumb_movement, right_thumb_movement)
        #     print("[THUMB] left: ", left_thumb_movement)
        #     print("[THUMB] right: ", right_thumb_movement)

        # if left_stable_activity in ['grab'] and right_thumb_change:
        #     right_thumb_movement = self.handManager.curr_right_stable_thumb_movement
        #     # user_not_moving = self.check_user_moving()
        #     # print(f"User is moving? {user_not_moving} \n" *100)
        #     if right_thumb_movement in ['move']: 
        #         print(f"right thumb: {right_thumb_movement}\n"*50)
        #         self.firebaseWriteManager.send_thumb_gesture({  "timestamp" : datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        #                                                         "timestamp_s": time.time(),
        #                                                         "right_thumb_change": right_thumb_change, 
        #                                                         "right_thumb_gesture": right_thumb_movement,
        #                                                         "left_thumb_change": left_thumb_change, 
        #                                                         "left_thumb_gesture": left_thumb_movement
        #                                                     })
        #         self.provide_reason_to_stop('turn')
        # elif right_stable_activity in ['grab'] and left_thumb_change:
        #     left_thumb_movement = self.handManager.curr_left_stable_thumb_movement
        #     # user_not_moving = self.check_user_moving()
        #     # print(f"User is moving? {user_not_moving} \n" *100)
        #     if left_thumb_movement in ['move']: 
        #         print(f"left thumb: {left_thumb_movement}\n"*50)
        #         self.firebaseWriteManager.send_thumb_gesture({  "timestamp" : datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        #                                                         "timestamp_s": time.time(),
        #                                                         "right_thumb_change": right_thumb_change, 
        #                                                         "right_thumb_gesture": right_thumb_movement,
        #                                                         "left_thumb_change": left_thumb_change, 
        #                                                         "left_thumb_gesture": left_thumb_movement
        #                                                     })
        #         self.provide_reason_to_stop('turn')

        

        annotated_frame = self.handManager.annotate_hands(annotated_frame, 
                                                            hands_results, 
                                                            gesture_data
                                                        )

        frame_info['current_caption'] = self.current_caption
        frame_info['current_caption_source'] = self.current_caption_source

        hand_activities = {
                "curr_left": self.handManager.curr_left_stable_activity if self.handManager.curr_left_stable_activity else "none",
                "curr_right": self.handManager.curr_right_stable_activity if self.handManager.curr_right_stable_activity else "none",

                "prev_left": self.handManager.prev_left_stable_activity if self.handManager.prev_left_stable_activity else "none",
                "prev_right": self.handManager.prev_right_stable_activity if self.handManager.prev_right_stable_activity else "none",

                "left_change": if_left_hand_change,
                "right_change": if_right_hand_change,

                "left_index_movement": left_index_movement,
                "right_index_movement": right_index_movement,

                "left_thumb_movement": left_thumb_movement,
                "right_thumb_movement": right_thumb_movement
            }
        
        if if_left_hand_change and self.handManager.curr_left_stable_activity in ['touch_explore', 'grab', 'pointer', None] or \
             if_right_hand_change and self.handManager.curr_right_stable_activity in ['touch_explore', 'grab', 'pointer', None]:
            
            if if_left_hand_change:
                if self.handManager.prev_left_stable_activity is None and self.handManager.curr_left_stable_activity is not None:
                    self.system_message.append({'text': "I see your left hand.",  
                                                'event':'see_hand', 
                                                'timestamp_s': time.time(), 
                                                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},)
                if self.handManager.curr_left_stable_activity is not None:
                    text = ""
                    text = "touching" if self.handManager.curr_left_stable_activity == 'touch_explore' else text
                    text = "grabbing" if self.handManager.curr_left_stable_activity == 'grab' else text
                    text = "pointing" if self.handManager.curr_left_stable_activity == 'pointer' else text
                    self.system_message.append({'text': f"left hand is {text}",  
                                                'event':'state_change', 
                                                'timestamp_s': time.time(), 
                                                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},)
                
                if self.handManager.curr_right_stable_activity is not None:
                    text = ""
                    text = "touching" if self.handManager.curr_right_stable_activity == 'touch_explore' else text
                    text = "grabbing" if self.handManager.curr_right_stable_activity == 'grab' else text
                    text = "pointing" if self.handManager.curr_right_stable_activity == 'pointer' else text
                    self.system_message.append({'text': f"right hand is {text}",  
                                                'event':'state_change', 
                                                'timestamp_s': time.time(), 
                                                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},)
            if if_right_hand_change: 
                if self.handManager.prev_right_stable_activity is None and self.handManager.curr_right_stable_activity is not None:
                    self.system_message.append({'text': "I see your right hand", 
                                                'event':'see_hand', 
                                                'timestamp_s': time.time(), 
                                                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")})
                
                if self.handManager.curr_right_stable_activity is not None:
                    text = ""
                    text = "touching" if self.handManager.curr_right_stable_activity == 'touch_explore' else text
                    text = "grabbing" if self.handManager.curr_right_stable_activity == 'grab' else text
                    text = "pointing" if self.handManager.curr_right_stable_activity == 'pointer' else text
                    self.system_message.append({'text': f"right hand is {text}",  
                                                'event':'state_change', 
                                                'timestamp_s': time.time(), 
                                                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},)
                
                if self.handManager.curr_left_stable_activity is not None:
                    text = ""
                    text = "touching" if self.handManager.curr_left_stable_activity == 'touch_explore' else text
                    text = "grabbing" if self.handManager.curr_left_stable_activity == 'grab' else text
                    text = "pointing" if self.handManager.curr_left_stable_activity == 'pointer' else text
                    self.system_message.append({'text': f"left hand is {text}",  
                                                'event':'state_change', 
                                                'timestamp_s': time.time(), 
                                                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},)

            self.provide_reason_to_stop("turn")    
            if if_left_hand_change and self.handManager.curr_left_stable_activity in ['touch_explore', 'grab', None] or \
             if_right_hand_change and self.handManager.curr_right_stable_activity in ['touch_explore', 'grab', None]:
            
                self.frame_history[self.frame_id] = frame_info
                # if self.stop_streaming_reason != 'text_interrupt':
                self.handscribe_for_new_scene(self.frame_id, frame, frame_info['uuid'], "new_scene", hand_activities)
                




        elif left_stable_activity == "pointer" and right_stable_activity in ["grab", "touch_explore"] or \
           right_stable_activity == "pointer" and left_stable_activity in ["grab", "touch_explore"]:
            # print(f"[STATE] pointing begins: left is {left_stable_activity} and right is {right_stable_activity}")
            hand_side = "left" if left_stable_activity == "pointer" else "right"
            the_other_hand_side = "right" if hand_side == "left" else "left"
            tip_location = self.handManager.get_index_finger_tip(frame, gesture_data, hand_side)
            hand_region = self.handManager.get_hand_region(frame, gesture_data, the_other_hand_side)
            if is_point_inside_bbox(tip_location, hand_region):
                # touched_text = detect_text_with_boxes(hand_frame, tip_location)
                # if not touched_text:
                touched_text = get_pointing_color(frame, hand_side, tip_location)
                
                annotated_frame = cv2.circle(annotated_frame, tip_location, 10, (0, 0, 255), -1)
                self.touched_text = {"touched_text": touched_text,
                                     'timestamp_s': time.time(), 
                                     'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}
                self.provide_reason_to_stop("color")
                print(f"[TOUCHED TEXT] {touched_text}")

        if not if_left_hand_change and self.handManager.curr_left_stable_activity in ['grab', 'touch_explore'] and self.handManager.curr_right_stable_activity != "pointer" or \
             not if_right_hand_change and self.handManager.curr_right_stable_activity in ['grab', 'touch_explore'] and self.handManager.curr_left_stable_activity != "pointer":
            # print(f"checking {self.handManager.curr_left_stable_activity} {self.handManager.curr_right_stable_activity}\n "*60)
            if self.new_object_check:
                self.new_object_check = False
                print("[INFO] sending images for checking")
                self.firebaseWriteManager.send_new_object_check_image({"timestamp_s"  : time.time(),
                                                                    "timestamp"    : datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                    "frame_base64" : encode_image_to_base64(self.current_frame)
                                                                    })
            
            if self.is_new_object_on_left or self.is_new_object_on_right:
                self.frame_history[self.frame_id] = frame_info
                hand_side = 'left' if self.is_new_object_on_left else 'right'
                hand_side = 'middle' if self.is_new_object_on_left and self.is_new_object_on_right else hand_side
                self.text_system_message.append({'text': f"I see you flip or change the {hand_side} object.",  
                                                'event':'state_change', 
                                                'timestamp_s': time.time(), 
                                                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},)
                self.provide_reason_to_stop("turn")

                hand_activities['left_change'] = self.is_new_object_on_left
                hand_activities['right_change'] = self.is_new_object_on_right

                self.handscribe_for_new_scene(self.frame_id, frame, frame_info['uuid'], "old_scene", hand_activities)
                if self.is_new_object_on_left: self.is_new_object_on_left = False
                if self.is_new_object_on_right: self.is_new_object_on_right = False
            
            
            # if self.handManager.is_index_movement_stable('left') or self.handManager.is_index_movement_stable('right'):
                
            #     if self.handManager.is_index_movement_stable('left') and self.handManager.curr_left_stable_index_movement == 'move' or \
            #        self.handManager.is_index_movement_stable('right') and self.handManager.curr_right_stable_index_movement == 'move': 
            #         print(f"{self.handManager.curr_left_stable_index_movement}, {self.handManager.curr_right_stable_index_movement} \n"*50)
            #         self.provide_reason_to_stop("turn")
            #         if self.handManager.is_index_movement_stable('left'): self.handManager.stable_left_index_activities.clear()
            #         if self.handManager.is_index_movement_stable('right'): self.handManager.stable_right_index_activities.clear()
                    
            #         self.frame_history[self.frame_id] = frame_info
            #         self.handscribe_for_new_scene(self.frame_id, frame, frame_info['uuid'], "new_scene", hand_activities)
            #         self.curr_scene_state["state"] = "new_scene"
            #         self.curr_scene_state["frame_id"] = self.frame_id
            #         self.curr_scene_state["uuid"] = frame_info["uuid"]
            #         self.curr_scene_state["sources_used"] = []
            #         self.curr_scene_state["descriptions"] = []

        self.tts_streaming()
        
        print(f"------------------ FPS: {self.frame_id / (time.time() - self.start_time)} ------------------")
        return annotated_frame

    def handscribe_for_new_scene(self, frame_id, frame, _uuid, state, hand_activities=None):

        self.prev_state_details = {
            "event"             : state,
            "uuid"              : _uuid,
            "frame_id"          : frame_id,
            "frame"             : frame,
            "ids"               : None,
            "clss"              : None,
            "frame"             : self.frame_history[frame_id]['frame'],
            
            "state_changed_time": time.time(),
            "timestamp"         : datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "timestamp_s"       : time.time(),
            "system_role"       : None,
            "user_msg"          : None,
            "user_goal"         : None,
            "user_goal_type"    : None,

            "sentence_requirement": None,
            "user_degree"         : self.user_current_degree,
            "hands_interact"      : 1, 
            "hand_activities"     : hand_activities
        }

        
        threading.Thread(target=self.generate_descriptions, args=(copy.deepcopy(self.prev_state_details),)).start()

        self.curr_scene_state["state"] = "new_scene"
        self.curr_scene_state["frame_id"] = self.frame_id
        self.curr_scene_state["uuid"] = _uuid
        self.curr_scene_state["sources_used"] = []
        self.curr_scene_state["descriptions"] = []

        if state == "old_scene": 
            print("old scene old scene old scene old scene\n" * 1000)
            self.curr_scene_state['sources_used'].append("moondream")

        return 
    

    def describe_for_new_scene(self, frame_id, _uuid, state, sentence_requirement, hand_activities=None):
        """
        Handles transitions to new caption states, including updates and server interactions.
        """
        
        print(f"[SENDING DESCRIPTION REQUEST]")
        print(f"[SENDING DESCRIPTION REQUEST]")
        print(f"[SENDING DESCRIPTION REQUEST]")
        print(f"[SENDING DESCRIPTION REQUEST]")

        system_role = self.user_preference.create_systemrole(sentence_requirement)
        user_msg = self.user_preference.create_user_requirement(sentence_requirement)

        if self.handManager.hands_interacting:
            system_role = self.user_preference.create_systemrole_for_interactions(sentence_requirement)
            user_msg = self.user_preference.create_user_requirement_for_interaction(sentence_requirement)
        
        state_changed_time = time.time()

        self.prev_state_details = {
            "event"             : state,
            "ids"               : self.ids_stable_state,
            "clss"              : self.frame_history[frame_id]['object_classes'],
            "frame"             : self.frame_history[frame_id]['frame'],
            "frame_id"          : frame_id,
            "state_changed_time": state_changed_time,
            "uuid"              : _uuid,
            "system_role"       : system_role,
            "user_msg"          : user_msg,
            "user_goal"         : self.user_preference.user_goal,
            "sentence_requirement": sentence_requirement,
            "user_degree": self.user_current_degree,
            "user_goal_type": self.user_preference.user_goal_type,
            "hands_interact": 1 if self.handManager.hands_interacting else 0
        }

        threading.Thread(target=self.generate_descriptions, args=(copy.deepcopy(self.prev_state_details),)).start()

    def update_frame_info_with_results(self, frame_info, results):
        """
        Updates the frame information dictionary with YOLO model results.
        """

        # print("self.user_preference.custom_classes: ", len(self.user_preference.custom_classes), self.user_preference.custom_classes)
        if results[0].boxes.id is not None:
            frame_info.update({
                "boxes": np.array(results[0].boxes.xyxy.cpu()),
                "ids": results[0].boxes.id.int().cpu().tolist(),
                "confs": results[0].boxes.conf.cpu().tolist(),
                "object_classes": [
                    self.user_preference.custom_classes[cls_num]
                    for cls_num in np.array(results[0].boxes.cls).astype(int)
                ]
            })

    def generate_caption_for_frame(self, frame_info, server_data_dict):
        """
        Generates a caption for the current frame and stores it in the frame history.
        """
        frame_info['yolo_caption'] = [{
            "caption": get_caption_from_yolo_classes_adv(frame_info, self.user_preference.object_preference),
            "frame_id": self.frame_id,
            "event": "default_caption",
            "state_changed_time": time.time(),
            "completion_time": time.time(),
            "uuid": frame_info["uuid"],
            "user_degree": self.user_current_degree,
            "payload": None,
            "ids": frame_info["ids"],
            "system_role": None,
            "user_msg": None,
            "user_goal": self.user_preference.user_goal,
            "sentence_requirement": None,
            "clss": frame_info["object_classes"],
            "similarity_score": None,
            "depth_score": None,
            "adj_preference": self.user_preference.adj_preference,
            "source": "yolo"
        }]
        self.frame_history[self.frame_id] = frame_info

    def annotate_frame(self, frame, detections):
        """
        Annotates a frame with bounding boxes and labels for visualization.
        """
        labels = [
            f"{self.user_preference.custom_classes[class_id]} {confidence:.3f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        return self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    
    def is_user_turn(self):
        
        if self.user_current_degree is None: 
            self.user_current_degree = self.firebaseWriteManager.get_user_degree()
        if self.user_prev_degree is None: 
            self.user_prev_degree = self.firebaseWriteManager.get_user_degree()
        
        diff = angle_difference(self.user_current_degree, self.user_prev_degree)

        if diff >= TURN_DEGREE_THRESHOLD:
            self.user_prev_degree = self.user_current_degree
            return True
        else: 
            return False


    def update_scene_state(self, frame_id, frame_uuid):
        if frame_id <= self.num_frame_to_check:
            return None, None
        return self.determine_scene_state(frame_id)

    def determine_scene_state(self, frame_id):
        """
        Determines the current scene state based on frame history and user actions.
        """
        
        recent_ids = self.get_recent_ids(frame_id, self.num_frame_to_check)
        # print(f"frame_id: {str(frame_id)}, recent_ids: {str(recent_ids)}")
        frame = self.frame_history[frame_id]['frame']

        # Initialize defaults.
        state = ""
        sentence_requirement = SENTENCE_LENGTH_10

        # state can be user_turn, new_scene, messy_scene, empty_scene
        if self.is_user_turn():
            state = "new_scene"

        elif self.is_new_scene(recent_ids, frame_id):
            # print("testing new scene....")
            state = self.handle_long_scene(recent_ids, frame_id, frame, "new_scene", "long_new_scene")

        elif self.is_empty_scene(recent_ids, frame_id):
            # print("testing empty scene....")
            state = self.handle_long_scene(recent_ids, frame_id, frame, "empty_scene", "long_empty_scene")

        if state and self.user_preference.granularity_preference['adaptive'] == "true":
            sentence_requirement = self.adapt_sentence_length(state, frame_id)
        elif state:
            sentence_requirement = self.get_sentence_length_preference()

        return state, sentence_requirement

    def generate_descriptions(self, prev_state_details):
        """
        Sends frame information to an external server for caption generation.
        """
        
        prev_state_details['frame'] = encode_image_to_base64(prev_state_details['frame'])
        # self.firebaseWriteManager.send_image_to_server(prev_state_details)
        self.firebaseWriteManager.send_image_to_handscribe_server(prev_state_details)
        
        return prev_state_details

    
    def has_n_repeated_elements(self, lst, n):
        """
        Check if any sublist in the list appears at least n times.

        Parameters:
            lst (list): The list to check, which can contain sublists.
            n (int): The number of repetitions to check for.

        Returns:
            tuple: (bool, element) where bool indicates if a repeated sublist exists, 
                and element is the repeated sublist (or None if not found).
        """
        # Convert sublists to tuples for hashing
        hashable_list = [tuple(sublist) if isinstance(sublist, list) else sublist for sublist in lst]
        
        # Count occurrences
        from collections import Counter
        element_counts = Counter(hashable_list)

        # Find an element repeated at least n times
        for element, count in element_counts.items():
            if count >= n:
                # Convert back to list if the element was originally a sublist
                element = list(element)
                element = sorted(element)
                return True, element

        return False, None
    
    def has_over_n_distinct_elements(self, input_list, N, m):
        """
        Checks if there are more than N distinct elements in the list, 
        where each distinct element appears more than m times.

        Args:
            input_list (list): A list of elements (elements may be lists).
            N (int): The threshold for the number of distinct elements.
            m (int): The minimum number of occurrences for an element to count as distinct.

        Returns:
            bool: True if there are more than N distinct elements that appear more than m times, False otherwise.
        """
        from collections import Counter

        # Convert all elements to tuples (if they are lists) to make them hashable
        hashable_elements = [tuple(elem) if isinstance(elem, list) else elem for elem in input_list]
        # Count occurrences of each element
        element_counts = Counter(hashable_elements)
        # Filter elements that appear more than m times
        frequent_elements = [elem for elem, count in element_counts.items() if count >= m]

        # Check if the number of distinct frequent elements exceeds N
        return len(frequent_elements) >= N

        
    def is_new_scene(self, recent_ids, frame_id):
        """
        Checks if the current scene is a new scene.
        """
        repeated , element = self.has_n_repeated_elements(recent_ids, len(recent_ids)-1) 
        return (
            repeated and len(element) > 0 and
            frame_id - self.prev_state_details['frame_id'] >= self.num_frame_to_check
        )

    def is_empty_scene(self, recent_ids, frame_id):
        """
        Checks if the current scene is an empty scene.
        """
        repeated , element = self.has_n_repeated_elements(recent_ids, len(recent_ids)-1) 
        return (
            repeated and len(element) == 0 and
            frame_id - self.prev_state_details['frame_id'] >= self.num_frame_to_check
        )


    def is_messy_scene(self, recent_ids, frame_id):
        """
        Checks if the current scene is a messy scene.
        """
        if self.has_over_n_distinct_elements(recent_ids, 2, 2) or self.has_over_n_distinct_elements(recent_ids, 4, 1):
            return True
        return False


    def handle_long_scene(self, recent_ids, frame_id, frame, scene_state, long_scene_state):
        """
        Handles both new and long scene detection logic.
        """
        # long_num_frame_to_check = 24
        recent_repeated , recent_element = self.has_n_repeated_elements(recent_ids, len(recent_ids)-2) 
        # long_ids = self.get_recent_ids(frame_id, long_num_frame_to_check)
        print(f"ids_stable_state: {self.ids_stable_state}, recent_element: {recent_element}")
        
        if self.prev_state_details["frame"] is not None and frame is not None:
            similarity = get_frame_similarity(self.prev_state_details["frame"], frame)
            print(f"{'$'*40} similarity score {similarity[0][0]} {'$'*40}")
            if similarity[0][0] > 0.8: return None

        if recent_repeated and recent_element != self.ids_stable_state:
            self.ids_stable_state = recent_element
            # if new_scene_state == "messy_scene": self.ids_stable_state = MESSY_ELEMENT
            return scene_state
        else:
            return None


    def adapt_sentence_length(self, state, frame_id):
        """
        Adapts sentence length based on scene state and user preference.
        """
        if 'long' in state:
            return SENTENCE_LENGTH_20
        elif len(self.frame_history[frame_id]['ids']) <= 5:
            return SENTENCE_LENGTH_10
        return SENTENCE_LENGTH_5

    def get_sentence_length_preference(self):
        """
        Returns the sentence length based on user preferences.
        """
        preference = self.user_preference.granularity_preference
        if preference['verbose'] == 'true':
            return SENTENCE_LENGTH_20
        elif preference['normal'] == 'true':
            return SENTENCE_LENGTH_10
        elif preference['concise'] == 'true':
            return SENTENCE_LENGTH_5
        return SENTENCE_LENGTH_10

    def get_recent_ids(self, frame_id, count):
        """
        Retrieves the recent IDs from frame history.
        """
        return [self.frame_history[i]['ids'] for i in range(frame_id - count, frame_id)] if frame_id - count > 0 else None


    def get_caption_for_tts(self):
        selected_source = None
        selected_caption_info = None

        print("***** searching caption source")
        if len(self.text_system_message)>0: 
            print("***** text_system_message")
            caption = self.text_system_message[0]['text']
            selected_source = "system_message"
            selected_caption_info = {"caption"      : caption, 
                                     "source"       : selected_source, 
                                     "frame_id"     : self.frame_id, 
                                     "frame_base64" : encode_image_to_base64(self.current_frame),
                                     "timestamp"    : self.text_system_message[0]['timestamp'],
                                     "timestamp_s"  : self.text_system_message[0]['timestamp_s'],
                                    }
            self.reset_all_text_buffer()
            self.text_system_message = []
            self.system_message = []

        # elif len(self.object_text_buffer) > 0:
        #     print("***** object_text_buffer")
        #     text_list = [item['text'] for item in self.object_text_buffer]
        #     output_text = " \n".join(text_list)
        #     if 'I did not see texts.' in output_text or 'I am still recognizing the text.' in output_text:
        #         caption = output_text
        #     else:
        #         caption = "texts on this object: " + output_text
        #     selected_source = "object_text"
        #     selected_caption_info = {"caption": caption, 
        #                                 "source": selected_source, 
        #                                 "frame_id": self.frame_id, 
        #                                 "frame_base64": encode_image_to_base64(self.current_frame),
        #                                 "timestamp": self.object_text_buffer[0]['timestamp'],
        #                                 "timestamp_s": self.object_text_buffer[0]['timestamp_s']
        #                             }
        #     self.reset_all_text_buffer()
        #     self.object_text_buffer = []
        #     self.system_message = []

        elif self.touched_text:
            print("***** touched color")
            selected_source = "color"
            selected_caption_info = {"caption": self.touched_text['touched_text'], 
                                     "source": "text", 
                                     "frame_id": self.frame_id,
                                     "frame_base64": encode_image_to_base64(self.current_frame),
                                     "timestamp": self.touched_text['timestamp'],
                                     "timestamp_s": self.touched_text['timestamp_s']
                                    }
            self.reset_all_text_buffer()
            self.touched_text = {}
            self.system_message = []


        elif len(self.system_message)>0: 
            print("***** system_message")
            text_list = [item['text'] for item in self.system_message]
            caption = synthesize_hand_caption(text_list)
            selected_source = "system_message"
            selected_caption_info = {"caption"      : caption, 
                                     "source"       : selected_source, 
                                     "frame_id"     : self.frame_id, 
                                     "frame_base64" : encode_image_to_base64(self.current_frame),
                                     "timestamp"    : self.system_message[0]['timestamp'],
                                     "timestamp_s"  : self.system_message[0]['timestamp_s'],
                                    }
            self.system_message = []
        
        
        
        # elif self.curr_scene_state["state"] == "new_scene":
        #     print("***** new scene")
        #     index = self.curr_scene_state['sources_used'].count('gpt')
        #     try:
        #         temp = self.firebaseWriteManager.get_caption_buffer()[index]
        #         if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
        #         # if not self.is_valid_caption_queue(temp):
        #             return None, selected_caption_info
        #         if temp['frame_id'] >= self.prev_state_details['frame_id']:
        #         # and temp['caption'] not in self.curr_scene_state['descriptions']:
        #             # temp['uuid'] not in self.gpt_spoken_uuid:

        #             print("***** gpt4")
        #             self.curr_scene_state['sources_used'].append("gpt")
        #             self.curr_scene_state['descriptions'].append(temp['caption'])

        #             selected_source = "gpt"
        #             selected_caption_info = temp
        #         # print(f"{'^'*40} {selected_caption_info.keys()}")                    
        #     except Exception as e:
        #         print("[ERROR] An error occurred when searching for gpt caption:", e)
        #         selected_source = None
        #         self.curr_scene_state['sources_used'].clear()

        #     if "gpt" not in self.curr_scene_state['sources_used']:
        #         temp = self.firebaseWriteManager.get_moondream_caption_buffer()[0]
                

        #         if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
        #         # if not self.is_valid_caption_queue(temp):
        #             return None, selected_caption_info
        #         print("***** new scene",  temp['frame_id'],  self.prev_state_details['frame_id'])
        #         if temp['frame_id'] >= self.prev_state_details['frame_id'] and \
        #             temp['caption'] not in self.curr_scene_state['descriptions']:
        #             # temp['uuid'] not in self.moondream_spoken_uuid:
        #             print("***** moondream")
        #             self.curr_scene_state['sources_used'].append("moondream")
        #             self.curr_scene_state['descriptions'].append(temp['caption'])
        #             selected_source = "moondream"
        #             selected_caption_info = temp
            
            
        

        elif self.curr_scene_state["state"] == "new_scene":
            print("***** new scene")
            if "moondream" not in self.curr_scene_state['sources_used']:
                temp = self.firebaseWriteManager.get_moondream_caption_buffer()[0]
                print("***** moondream")
                

                if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
                # if not self.is_valid_caption_queue(temp):
                    return None, selected_caption_info
                print("***** new scene",  temp['frame_id'],  self.prev_state_details['frame_id'])
                if temp['frame_id'] >= self.prev_state_details['frame_id'] and \
                    temp['caption'] not in self.curr_scene_state['descriptions']:
                    # temp['uuid'] not in self.moondream_spoken_uuid:
                    print("***** moondream")
                    self.curr_scene_state['sources_used'].append("moondream")
                    self.curr_scene_state['descriptions'].append(temp['caption'])
                    selected_source = "moondream"
                    selected_caption_info = temp
            
            elif "moondream" in self.curr_scene_state['sources_used']:
                
                index = self.curr_scene_state['sources_used'].count('gpt')
                try:
                    print("***** gpt4")
                    temp = self.firebaseWriteManager.get_caption_buffer()[index]
                    if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
                    # if not self.is_valid_caption_queue(temp):
                        return None, selected_caption_info
                    if temp['frame_id'] >= self.prev_state_details['frame_id']:
                        # temp['caption'] not in self.curr_scene_state['descriptions']:
                        # temp['uuid'] not in self.gpt_spoken_uuid:

                        print("***** gpt4 2")
                        self.curr_scene_state['sources_used'].append("gpt")
                        self.curr_scene_state['descriptions'].append(temp['caption'])

                        selected_source = "gpt"
                        selected_caption_info = temp
                    # print(f"{'^'*40} {selected_caption_info.keys()}")                    
                except Exception as e:
                    print("[ERROR] An error occurred when searching for gpt caption:", e)
                    selected_source = None
                    self.curr_scene_state['sources_used'].clear()
                    
        # Fallback to YOLO captions if no other caption queue is selected
        if not selected_source:
            print("[INFO] No valid caption found.")
            return None, selected_caption_info # uncomment here 
        
        # Validate the selected caption
        if not selected_caption_info or isinstance(selected_caption_info,list) or isinstance(selected_caption_info,str):
            print("[INFO] No valid caption found.")
            return None, selected_caption_info

        
        caption  = selected_caption_info["caption"]
        # source   = selected_caption_info["source"]
        # uuid     = selected_caption_info["uuid"]
        # frame_id = selected_caption_info["frame_id"]

        self.current_caption_source = selected_source
        self.current_caption = caption
        
        self.curr_scene_state['descriptions'].append(caption)


        print(f"{'^'*100}")
        print(f"[CAPTION]: {caption}")
        print(f"{'^'*100}")

        # self.firestoreManager.read_caption_update(selected_caption_info)

        return caption, selected_caption_info

    def select_best_caption(self, caption_queues):
        """
        Selects the best caption from available caption queues based on similarity and state.
        """
        for queue_name, caption_queue in caption_queues.items():
            if not self.is_valid_caption_queue(caption_queue):
                continue

            caption_info = caption_queue[0]

            # Check UUID and other attributes for selection criteria
            if self.is_matching_uuid(caption_info) or self.is_similar_to_previous_frame(caption_info):
                # print(f"\n\nSelected caption from {queue_name}. \n\n{caption_info}")
                return queue_name, caption_info

        return None, None

    def is_valid_caption_queue(self, caption_queue):
        """
        Validates the caption queue to ensure it contains valid data.
        """
        return caption_queue not in [None, "start", ["s", "t", "a", "r", "t"]]

    def is_matching_uuid(self, caption):
        """
        Checks if the caption UUID matches the previous state UUID.
        """
        return caption.get("uuid") == self.prev_state_details["uuid"]

    def is_similar_to_previous_frame(self, caption):
        """
        Checks if the caption corresponds to a frame similar to the previous state frame.
        """
        similarity = get_frame_similarity(
            self.prev_state_details["frame"],
            self.frame_history[caption["frame_id"]]["frame"]
        )
        return similarity > SIMILARITY_THRESHOLD

    def manipulate_content(self, caption):
        """
        Modifies the caption content based on user preferences and syntax corrections.
        """
        if not caption:
            return None

        if self.is_talking_less:
            caption = concise_sentence(caption)

        return caption.replace(" ,", ",").replace(" .", ".").replace("  ", " ")


    def filter_caption_queue(self, caption_queue):
        """
        Checking the previous sentences and avoiding repeating
        """
        if len(caption_queue) == 0: return None
        if caption_queue[0]['source'] == "yolo": 
            return caption_queue[0]

        caption_history = [ (item['caption'],item['uuid']) for item in self.caption_history]
        
        d = 3 # set the number of previous descriptions to check
        index = len(self.caption_history) if len(self.caption_history) < d else d
        caption_latest = [item['caption'] for item in self.caption_history[-index:]]

        for item in caption_queue:
            if (item['caption'], item['uuid']) not in caption_history:            
                if item['caption'] not in caption_latest:
                    if len(self.caption_history) < d:
                        self.caption_history.append(item)
                        return item
                    else:
                        scores = get_sentence_similarity_spacy([item['caption']], caption_latest)
                        if max(scores) < 0.8:
                            self.caption_history.append(item)
                            return item
        return None
    
    def provide_reason_to_stop(self, reason):
        
        dont_interrupt = [
            "invoke_agent",
            "waiting_for_user_query_and_answer",
            "text_interrupt",
            "text_reading"
        ]
        if self.stop_streaming_reason != reason:
            if reason == "invoke_agent":
                self.stop_streaming_reason = reason
                return "invoke_agent"
            elif reason == "color":
                self.stop_streaming_reason = reason
                return "color"
            elif reason == "text_interrupt":
                self.stop_streaming_reason = reason
                return "text_interrupt"
            
            elif reason == "turn" and self.stop_streaming_reason not in dont_interrupt and not self.current_caption_source == "system_message":
                self.stop_streaming_reason = reason
                return "turn"
    
    def reset_all_text_buffer(self):
        self.system_message = []
        self.text_system_message = []
        self.object_text_buffer = []
        self.touched_text = {}
        return
    

    def tts_streaming(self):
        """
        Streaming the descriptions by tts service from OpenAI. Check it out at speech.py
        """
        try:
            caption = None
            # print(f"self.current_caption_source: {self.current_caption_source} \n"*100 )
            # if self.stop_streaming_reason == "text_interrupt":
            #     self.ttsManager.stop_streaming()
            #     print("***turn interrupt***\n" * 30)
            #     self.firebaseWriteManager.send_stop_streaming("turn")
            #     # self.ttsManager.lock_streaming()
            #     self.stop_streaming_reason = ""
            #     self.ttsManager.is_streaming = False

            if self.stop_streaming_reason == "text_interrupt":
                self.ttsManager.stop_streaming()
                self.ttsManager.lock_streaming()
                self.firebaseWriteManager.send_stop_streaming("invoke_text")
                self.stop_streaming_reason = "text_reading"
            
            elif self.stop_streaming_reason == "text_reading":
                if len(self.object_text_buffer) > 0:
                    print("***** object_text_buffer")
                    text_list = [item['text'] for item in self.object_text_buffer]
                    output_text = " \n".join(text_list)
                    if 'I did not see texts.' in output_text or 'I am still recognizing the text' in output_text:
                        caption = output_text
                    else:
                        caption = "texts on this object: " + output_text
                    selected_source = "object_text"
                    selected_caption_info = {"caption": caption, 
                                                "source": selected_source, 
                                                "frame_id": self.frame_id, 
                                                "frame_base64": encode_image_to_base64(self.current_frame),
                                                "timestamp": self.object_text_buffer[0]['timestamp'],
                                                "timestamp_s": self.object_text_buffer[0]['timestamp_s'],
                                            }
                    selected_caption_info['spoken_timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    selected_caption_info['spoken_timestamp_s'] = time.time()

                    print(f"caption of object text is {caption} \n"*20)
                    self.firebaseWriteManager.send_memory_image_to_server(selected_caption_info)
                    self.firebaseWriteManager.send_text_reading(caption)
                    self.stop_streaming_reason = ""
                    self.reset_all_text_buffer()
                    return
                return


            elif self.stop_streaming_reason == "turn":
                # print(f"self.stop_streaming_reason: {self.stop_streaming_reason} \n"*100)
                if self.current_caption_source == "system_message" or self.current_caption_source == "object_text":
                    self.stop_streaming_reason = ""
                    return
                elif self.force_to_read:
                    sentences = [item['text'] for item in self.system_message]
                    caption = synthesize_hand_caption(sentences)
                    self.firebaseWriteManager.send_agent_ending_msg(caption, 'ending')
                    self.stop_streaming_reason = ""
                    self.force_to_read = False
                    self.ttsManager.lock_streaming()
                    self.ttsManager.is_streaming = True
                    return
                
                self.ttsManager.stop_streaming()
                print("***turn interrupt***\n" * 30)
                self.firebaseWriteManager.send_stop_streaming("turn")
                self.stop_streaming_reason = ""
                self.ttsManager.is_streaming = False
                return

            elif self.stop_streaming_reason == "color":
                if not self.current_caption_source == "color":
                    self.ttsManager.stop_streaming()
                    print("***color interrupt***")
                    self.firebaseWriteManager.send_stop_streaming("turn")
                    self.ttsManager.is_streaming = False
                    
            
            elif self.stop_streaming_reason == "invoke_agent":
                self.ttsManager.stop_streaming()
                self.ttsManager.lock_streaming()
                
                self.firebaseWriteManager.send_stop_streaming("invoke_agent")
                self.stop_streaming_reason = "waiting_for_user_query_and_answer"
                return
            
            elif self.stop_streaming_reason == "waiting_for_user_query_and_answer":
                print("waiting_for_user_query_and_answer")
                if self.user_preference.agent_response:
                    print("waiting_for_user_query_and_answer2")
                    caption = self.user_preference.agent_response
                    print(f"agent response: {caption}"*60)

                    self.firebaseWriteManager.send_agent_response(caption, 'agent_response')
                    self.reset_all_text_buffer()
                    
                    self.stop_streaming_reason = ""
                    self.user_preference.agent_response = None                    
                    self.ttsManager.is_streaming = True
                    return
        

            print(f"is_streaming: {self.ttsManager.is_streaming}, stream_locked: {self.ttsManager.stream_locked} \n" *10)
            if not self.ttsManager.is_streaming and not self.ttsManager.stream_locked:
                if not caption:
                    caption, selected_caption_info = self.get_caption_for_tts()
                    
                if caption == None or caption == "": 
                    self.ttsManager.is_streaming = False
                    return
                # if selected_caption_info['source'] != 'text':
                # selected_caption_info['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                # selected_caption_info['timestamp_s'] = time.time()
                
                selected_caption_info['spoken_timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                selected_caption_info['spoken_timestamp_s'] = time.time()
                
                self.firebaseWriteManager.send_memory_image_to_server(selected_caption_info)

                    # if selected_caption_info['source'] == 'moondream': self.gpt_spoken_uuid.append(selected_caption_info['uuid'])
                    # elif selected_caption_info['source'] == 'gpt4': self.gpt_spoken_uuid.append(selected_caption_info['uuid'])
                    
                # self.current_caption =  caption
                self.ttsManager.is_streaming = True
                self.firebaseWriteManager.update_caption(caption)
                return

        except queue.Empty:
            pass

if __name__ == '__main__':
    import torch
    print(torch.backends.mps.is_available())
