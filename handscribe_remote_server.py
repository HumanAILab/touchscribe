import uuid
import traceback
import json
from collections import deque
import ast
import time
import base64
import torch
import requests
import threading
import json
import base64
import cv2
import os
from PIL import Image
from datetime import datetime
import gzip
import base64

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np
# Initialize the app with a service account
if not firebase_admin._apps:
    cred = credentials.Certificate(os.environ.get('FIREBASE_CREDENTIALS_PATH', 'firebase-adminsdk.json'))
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.environ.get('FIREBASE_DATABASE_URL', 'https://your-project-default-rtdb.firebaseio.com/')
    })
from utils.firebase.firestore_manager import FirestoreManager

from utils.gpt4v import headers, process_response, prepare_inputs
# from utils.point_cloud_utils import save_colorized_point_cloud
from utils.worldscribe_utils import base64_to_cv2_image, encode_image_to_base64
from utils.firebase.firebase_manager import FirebaseWriteManager
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.hands23.hands23 import HandsDetector
from memory_manager import MemoryManager
from utils.embedding_encoder import encode_image, encode_text, cosine_similarity
from utils.increase_res_and_get_text import increase_resolution, visualize_easyocr, get_object_text

torch.cuda.empty_cache()
moondream_device = "cuda:1" # TODO: change 1 and 0

device1 = torch.device(moondream_device if torch.cuda.is_available() else "cpu")

model_id = "vikhyatk/moondream2"
# revision = "2024-08-26"
revision = "2025-01-09"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
    # , attn_implementation="flash_attention_2"
).to(device1, torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

moondream_prompt_both_grab = "What are my hands grabbing?"
moondream_prompt_single_grab = "What is my hand grabbing?"
moondream_prompt_both_touch_explore = "What are my hands touching?"
moondream_prompt_single_touch_explore = "What is my hand touching?"

def count_numbers_in_range(numbers, lower=0, upper=0.9):
    """
    Count how many numbers in the list are between `lower` and `upper` (inclusive of lower, exclusive of upper).

    :param numbers: List of numbers.
    :param lower: Lower bound (inclusive).
    :param upper: Upper bound (exclusive).
    :return: Count of numbers within the range.
    """
    return sum(1 for num in numbers if lower <= num < upper)

class RemoteServer:

    def __init__(self):
        self.memoryManager = MemoryManager()
        self.firestoreManager = FirestoreManager()
        self.handsDetector = HandsDetector()
        
        self.current_thread = None
        self.stop_event = threading.Event()
        self.folder_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.ref_hand_object_image_start = True
        self.ref_handscribe_image_data = db.reference('ref_handscribe_image_data/')
        self.ref_handscribe_image_data.listen(self.listener_hand_object_image_data)
        
        self.ref_memory_image_data_start = True
        self.ref_memory_image_data = db.reference('memory_image_data/')
        self.ref_memory_image_data.listen(self.listener_memory_image_data)
        
        self.ref_moondream = db.reference('moondream_buffer/')
        
        self.caption_history = deque(maxlen=40)
        
        self.firebaseWriteManager = FirebaseWriteManager()
        print(f"[INFO] REMOTE SERVER STARTS TO RECEIVING TASKS...")
    
        # self.ref_user_name_start = True
        # self.ref_user_name = db.reference('user_name/')
        # self.ref_user_name.listen(self.listen_user_name)
        
        
        self.ref_swipe_data_start = True
        self.ref_swipe_data = db.reference('swipe_data/')
        self.ref_swipe_data.listen(self.listener_swipe_data)
        
        self.ref_object_text = db.reference('object_text/')
        
        self.ref_user_query_start = True
        self.ref_user_query = db.reference('user_query_buffer/')
        self.ref_user_query.listen(self.listener_user_query)
        
        self.ref_user_query_answer = db.reference('user_query_answer/')
        
        self.ref_text_system_message = db.reference('text_system_message/')
        
        
        
        self.ref_new_object_check_image_start = True
        self.ref_new_object_check_image = db.reference('new_object_check_image/')
        self.ref_new_object_check_image.listen(self.listener_new_object_check_image)
        self.ref_new_object_check = db.reference('new_object_check/')
        
        
        self.image_history_length = 5
        self.right_hand_image_history = deque(maxlen=int(self.image_history_length))
        self.left_hand_image_history = deque(maxlen=int(self.image_history_length))
        
        self.moondream_is_working = False
        
        
        self.left_object_text = []
        self.right_object_text = []
        
        # self.object_text = []
        # self.object_text_side = None
        self.object_text_index = 0
        self.current_frame_id = 0
        
    def is_not_current_frame(self, frame_id, function_name=""):
        if frame_id != self.current_frame_id:
            print(f"{function_name}; current is {self.current_frame_id}, frame {frame_id} is outdated")
        return frame_id != self.current_frame_id
        
    def listener_new_object_check_image(self, packet):
        if self.ref_new_object_check_image_start:
            self.ref_new_object_check_image_start = False
            print("******************************")
            print("Listening to new object check image")
            print("******************************")
            return
        is_new_object_on_left = False
        is_new_object_on_right = False
        
        threshold = 0.85
        
        packet = dict(packet.data)
        frame_base64 = packet['frame_base64']
        frame_cv2 = base64_to_cv2_image(frame_base64)
        frame_cv2 = cv2.rotate(frame_cv2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        hand_interaction_info = self.handsDetector.detect_hands(frame_cv2)
        
        for hand in hand_interaction_info['which_hands']:
            if hand_interaction_info[hand]['contactState'] != "no_contact":
                embedding = encode_image(hand_interaction_info[hand]['obj_image']) if hand_interaction_info[hand]['obj_image'] is not None else None
                if hand == 'right': self.right_hand_image_history.append(embedding)
                elif hand == 'left': self.left_hand_image_history.append(embedding)
                print(f"******embedding {hand} checking*******")
        
        if len(self.right_hand_image_history) == self.image_history_length:
            scores = cosine_similarity(list(self.right_hand_image_history)[:-1], list(self.right_hand_image_history)[-1])
            num = count_numbers_in_range(scores, lower=0, upper=threshold)
            is_new_object_on_right = num >= len(list(self.right_hand_image_history))-1
            if is_new_object_on_right: self.right_hand_image_history.clear()
            print("[RIGHT SCORES]", is_new_object_on_right, scores)
            
        if len(self.left_hand_image_history) == self.image_history_length:
            scores  = cosine_similarity(list(self.left_hand_image_history)[:-1], list(self.left_hand_image_history)[-1])
            num = count_numbers_in_range(scores, lower=0, upper=threshold)
            is_new_object_on_left = num >= len(list(self.left_hand_image_history))-1
            if is_new_object_on_left: self.left_hand_image_history.clear()
            print("[LEFT SCORES]", is_new_object_on_left, scores)
        
        self.ref_new_object_check.set({"timestamp_s"            : time.time(),
                                       "timestamp"              : packet['timestamp'],
                                       "complete"               : True,
                                       "is_new_object_on_left"  : is_new_object_on_left,
                                       "is_new_object_on_right" : is_new_object_on_right,
                                       })
        
        return 
    
    
    def get_text_for_the_object(self, image_base64, frame_id, hand_side):
        # image_cv2 = base64_to_cv2_image(image_base64)
        # image_base64 = encode_image_to_base64(image_cv2) 
        
        print("######################")
        print(f"{hand_side} getting text from the object")
        print(f"{hand_side} getting text from the object")
        print(f"{hand_side} getting text from the object")
        print(f"{hand_side} getting text from the object")
        print("######################\n")
        
        if self.is_not_current_frame(frame_id, "get_text_for_the_object"): return
        image_cv2 = base64_to_cv2_image(image_base64)
        image_cv2 = cv2.rotate(image_cv2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        # image_cv2 = increase_resolution(image_cv2)
        cv2.imwrite('increased_resolution.jpg', image_cv2)
        
        image_base64 = encode_image_to_base64(image_cv2)
        if self.is_not_current_frame(frame_id, "get_text_for_the_object"): return
        
        
        systemrole = "Your role is to describe the text on the object word by word."
        user_msg = f"I am holding the object with my {hand_side} hand. Please describe the text line by line and use semicolon to separate each word. Return only the text. don't contain '\n'. If there is no text, can you just return 'no text on the [object name] in your [hand side] hand' "
        messages = [{"role": "developer", "content": [systemrole]}]   
        messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }) 
        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 10000,
            "temperature": 0
        }
        
        # payload = prepare_inputs(systemrole, image_base64, user_msg)
        if self.is_not_current_frame(frame_id, "get_text_for_the_object"): return
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        res = response.json()['choices'][0]['message']['content']
        words = []
        if self.is_not_current_frame(frame_id, "get_text_for_the_object"): return
        
        print("######################")
        print("this is the text:", res)
        print("######################\n")
        try: 
            if 'sorry' in res.lower() or "can't" in res.lower() or 'unable' in res.lower() or 'transcribe' in res.lower():
                if self.is_not_current_frame(frame_id, "get_text_for_the_object"): return
                self.send_failed_text_message("Text detection failed or no text on this object. Please try again.")
                return 
            
            if 'no text' in res.lower():
                print("no text condition", res)
                if self.is_not_current_frame(frame_id, "get_text_for_the_object"): return
                words = res.split(';')
                words = [word.replace('\n', '') for word in words if word != '']
                words = [' '.join(words)]
                self.send_failed_text_message(f"{words}, please try another angle.")
                return
            words = res.split(';')
            words = [word.replace('\n', '') for word in words if word != '']
            words = [' '.join(words)]
            print(f"[{hand_side} WORDS FROM GPT]", words)
            print(f"[{hand_side} WORDS FROM GPT]", words)
            print(f"[{hand_side} WORDS FROM GPT]", words)
            
            
            if 'no text' in words[0].lower():
                print("no text condition", res)
                if self.is_not_current_frame(frame_id, "get_text_for_the_object"): return
                self.send_failed_text_message(f"{res}, please try another angle.")
                return
            
            if not words: 
                if self.is_not_current_frame(frame_id, "get_text_for_the_object"): return
                self.send_failed_text_message("No text on the object. Please try another angle.")
                return
            if self.is_not_current_frame(frame_id, "get_text_for_the_object"): return
            
            if hand_side == 'right':
                self.right_object_text = words
            elif hand_side == 'left':
                self.left_object_text = words
            else:
                self.right_object_text = words
                self.left_object_text = words
            
            self.send_successful_text_message(hand_side, words)            
        except:
            print("errors happened in get_keywords_for_images")
            self.send_failed_text_message()
        return words

    def send_failed_text_message(self, text):
        self.ref_text_system_message.set({
                    "timestamp_s": time.time(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text": text
                })
    def send_successful_text_message(self, hand_side, words):
        if hand_side == 'both':
            text = f"This is text on the object: \n\n" + words[0]
        else:    
            text = f"This is text on the {hand_side} object: \n\n" + words[0]
        self.ref_text_system_message.set({
                    "timestamp_s": time.time(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text": text
                })
    
    def listener_swipe_data(self, packet):
        if self.ref_swipe_data_start:
            self.ref_swipe_data_start = False
            print("******************************")
            print("Listening to thumb swipe data")
            print("******************************")
            return
        
        try:
            swipe_data = dict(packet.data)
            left_object_change  = swipe_data['left_object_change']
            right_object_change = swipe_data['right_object_change']
            
            if left_object_change:
                output = ""
                if len(self.left_object_text) == 0:
                    output = "I am still recognizing the text of the left object. Please try again in a few seconds."
                else:
                    output = self.left_object_text[self.object_text_index]
                self.ref_object_text.set({"text": output, "timestamp": swipe_data['timestamp'], "timestamp_s": swipe_data['timestamp_s']})
                print(f"[SWIPE TO GET OBJECT TEXT] {output} \n "*5)
            elif right_object_change:
                output = ""
                if len(self.right_object_text) == 0:
                    output = "I am still recognizing the text of the right object. Please try again in a few seconds."
                else:
                    output = self.right_object_text[self.object_text_index]
                self.ref_object_text.set({"text": output, "timestamp": swipe_data['timestamp'], "timestamp_s": swipe_data['timestamp_s']})
                print(f"[SWIPE TO GET OBJECT TEXT] {output} \n "*5)
        except Exception as e:
            print("Error: ", e)
            self.object_text_index = 0
        return output
    
    
    def listener_user_query(self, event):
        try:
            if self.ref_user_query_start:
                self.ref_user_query_start = False
            else:
                
                packet = dict(event.data)
                frame_base64 = packet['frame_base64']
                user_query   = packet['user_query']
                timestamp_s  = packet['timestamp_s']                
                
                
                print(f"[USER QUERY] {user_query}\n"*5)

                
                history_systemrole="""
                You are a visual describer, you need to examine the previous decriptions and images you provided to the user and answer their questions with previous contexts (if any). 
                """
                
                messages = [{"role": "system", "content": [history_systemrole]}]
                # history = self.memoryManager.single_search_with_img_embeddings(frame_base64, '2d_object', "full_image_embedding", 10 , 5)
                # for idx, (hist_image, hist_description, hist_date, hist_seconds, left_gesture, right_gesture, hand_side) in enumerate(history):  # Adjust number as needed
                #     history_entry =  f"Here is the previous description and the image you described to the user at {hist_date}.\n"
                #     seconds_ago = int(time.time()) - hist_seconds
                #     minutes = seconds_ago // 60
                #     seconds = seconds_ago % 60

                #     # Format as string
                #     time_string = f"{minutes} minutes" if minutes > 0 else f"{seconds} seconds"

                #     history_entry += f"History {idx+1} ({time_string} ago): {hist_description}. User used {hand_side} hands to interact with this object, by"
                #     if left_gesture != "none": history_entry += f" {left_gesture} the object with {hand_side} hand."
                #     if right_gesture != "none": history_entry += f" {right_gesture} the object with {hand_side} hand."

                #     messages.append({
                #         "role": "user",
                #         "content": [
                #             {"type": "text", "text": history_entry},
                #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{hist_image}"}}
                #         ]
                #     })

                # user_msg="Can you describe if I accessed this object before. How long ago and what was I doing? If it is a new object, describe it in detail."
                
                # Add the current query
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}}
                    ]
                })
            
                payload = {
                    "model": "gpt-4o",
                    "messages": messages,
                    "max_tokens": 10000,
                    "temperature": 0.1
                }
                
                
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
                res = response.json()['choices'][0]['message']['content']
                
                
                data = {
                        'response': res,
                        'timestamp_s': time.time()
                        }
                
                self.ref_user_query_answer.set(data)
                
                
                
        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback

        return
    
    def listener_hand_object_image_data(self, packet):
        if self.ref_hand_object_image_start:
            self.ref_hand_object_image_start = False
            print("******************************")
            print("Listening to hand-object interaction info")
            print("******************************")
            return

        # Cancel the previous thread if it exists
        if self.current_thread and self.current_thread.is_alive():
            print("[INFO] Stopping previous processing thread")
            self.stop_event.set()
            self.current_thread.join()

        # Reset the stop event for the new thread
        self.stop_event.clear()

        # Start a new thread for processing
        self.current_frame_id =  dict(packet.data)['frame_id']
        self.current_thread = threading.Thread(target=self.process_packet_data, args=(packet,))
        self.current_thread.start()
    
    def process_packet_data(self, packet):
        packet = dict(packet.data)
        start_time = time.time()
        # try:
        image_base64              = packet['frame']
        frame_cv2                 = base64_to_cv2_image(image_base64)
        frame_cv2                 = cv2.rotate(frame_cv2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_id                  = packet['frame_id']
        uuid                      = packet['uuid']
        hand_activities_mediapipe = packet['hand_activities']
        
        
        # self.object_text = []
        
        
        # print("hand_activities_mediapipe", hand_activities_mediapipe)

        right_change = hand_activities_mediapipe['right_change']
        left_change  = hand_activities_mediapipe['left_change']
        curr_left    = hand_activities_mediapipe['curr_left'] if 'curr_left' != 'none' else None
        curr_right   = hand_activities_mediapipe['curr_right'] if 'curr_right' != 'none' else None  
        prev_left    = hand_activities_mediapipe['prev_left'] if 'prev_left' != 'none' else None
        prev_right   = hand_activities_mediapipe['prev_right'] if 'prev_right' != 'none' else None


        if right_change: self.right_hand_image_history.clear()
        if left_change: self.left_hand_image_history.clear()

        if curr_left is None or curr_left == "pointer":
            self.left_object_text = []
        if curr_right is None or curr_right == "pointer":
            self.right_object_text = []

        
        hand_start_time = time.time()
        hand_interaction_info     = self.handsDetector.detect_hands(frame_cv2)
        # print(f"[HAND DETECTION] takes {time.time() - hand_start_time} seconds")
        # print(f"[HAND DETECTION] takes {time.time() - hand_start_time} seconds")
        # print(f"[HAND DETECTION] takes {time.time() - hand_start_time} seconds")
        num_of_hands              = hand_interaction_info['num_of_hands']
        interact_with_same_object = hand_interaction_info['interact_with_same_object']

        if self.is_not_current_frame(frame_id): return


        hand_activities = hand_activities_mediapipe
        right_obj_mask_base64  = None
        left_obj_mask_base64   = None
        right_obj_image_base64 = None
        left_obj_image_base64  = None
        moondream_context = []
        
        def process_hand_caption(hand_side, obj_image, contact, prompt):
            """Processes a caption for a given hand."""
            if obj_image is None or contact == "no_contact":
                print(f"[ERROR] No object is being touched in {hand_side}")
                return None
            # moondream_time = time.time()
            caption = process_response(self.moondream_inference_unimanual(packet, obj_image, hand_side, prompt))
            # print(f"[MOONDREAM] {time.time() - moondream_time} seconds\n"*5)
            if caption is None:
                print(f"[ERROR] No moondream caption is generated for {hand_side}")
            return caption

        def process_single_hand(hand_side):
            """Processes a single-hand interaction."""
            nonlocal right_obj_mask_base64, left_obj_mask_base64, right_obj_image_base64, left_obj_image_base64
            hand_data = hand_interaction_info[hand_side]
            print(f"[INFO] Processing single {hand_side} hand - Contact State: {hand_data['contactState']}")
            
            if hand_data['contactState'] == "no_contact":
                return
            
            if hand_side == 'right':
                
                right_obj_mask_base64 = encode_image_to_base64(hand_data['obj_masks'])
                right_obj_image_base64 = encode_image_to_base64(hand_data['obj_image'])
                print("[ENCODED] right_obj_mask_base64")
            else:
                left_obj_mask_base64 = encode_image_to_base64(hand_data['obj_masks'])
                left_obj_image_base64 = encode_image_to_base64(hand_data['obj_image'])
                print("[ENCODED] left_obj_image_base64")
            
            if self.is_not_current_frame(frame_id): return
            caption = process_hand_caption(hand_side, hand_data['obj_image'], hand_data['contactState'], f"What object is my {hand_side} hand touching?")
            if self.is_not_current_frame(frame_id): return

            if caption:
                obj_image_base64 = right_obj_image_base64 if hand_side == 'right' else left_obj_image_base64
                hand_data['caption'] = caption
                # if hand_side == 'right': 
                threading.Thread(target=self.get_text_for_the_object, args=(obj_image_base64, frame_id, hand_side, )).start()                
                if self.is_not_current_frame(frame_id): return
                moondream_context.append((obj_image_base64, caption, hand_side, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

        def process_both_hands_same_object(custom_prompt, left_gesture=None, right_gesture=None):
            """Handles cases where both hands are involved."""
            nonlocal right_obj_mask_base64, left_obj_mask_base64, right_obj_image_base64, left_obj_image_base64
            
            left_hand, right_hand = hand_interaction_info['left'], hand_interaction_info['right']
            has_image = left_hand['obj_image'] is None and right_hand['obj_image'] is None

            if has_image or (left_hand['contactState'] == "no_contact" and right_hand['contactState'] == "no_contact"):
                print("[ERROR] No object is being touched")
                return
            
            right_obj_image_base64 = encode_image_to_base64(right_hand['obj_image'])
            right_obj_mask_base64  = encode_image_to_base64(right_hand['obj_masks'])
            
            left_obj_image_base64  = encode_image_to_base64(left_hand['obj_image']) 
            left_obj_mask_base64   = encode_image_to_base64(left_hand['obj_masks']) 
            obj_image = left_hand['obj_image'] if left_hand['obj_image'] is not None else right_hand['obj_image']
            # obj_mask_base64 = encode_image_to_base64(left_hand['obj_masks'] if left_hand['obj_masks'] is not None else right_hand['obj_masks'])

            prompt = custom_prompt
            # prompt = "What object are my hands touching separately?" if curr_left == "touch_explore" and curr_right == "touch_explore" else "What object are my hands touching?"
            if self.is_not_current_frame(frame_id): return
            caption = process_hand_caption("both", obj_image, "contact", prompt)
            if self.is_not_current_frame(frame_id): return
            
            
            if caption:
                left_hand['caption'] = right_hand['caption'] = caption
                # obj_image_base64 = encode_image_to_base64(obj_image)
                threading.Thread(target=self.get_text_for_the_object, args=(right_obj_image_base64, frame_id, "right")).start()
                threading.Thread(target=self.get_text_for_the_object, args=(left_obj_image_base64, frame_id, "left")).start()
                if self.is_not_current_frame(frame_id): return
                moondream_context.append((right_obj_image_base64, caption, "both", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

        # Determine the interaction case
        hand_count = len(hand_interaction_info['which_hands'])

        hand_side = ""
        if hand_count == 2:
            print("[HAND] Processing both hands")
            hand_side = 'both'
            if self.is_not_current_frame(frame_id): return

            if curr_left == "touch_explore" and curr_right == "touch_explore" and not interact_with_same_object:
                print("[HAND] Touch exploring with both hands")
                custom_prompt = "What is the object I am touching?"
                if self.is_not_current_frame(frame_id): return
                process_single_hand('left')
                if self.is_not_current_frame(frame_id): return
                process_single_hand('right')
                # process_both_hands(custom_prompt, curr_left, curr_right)
                
                
            elif curr_left == "grab" and curr_right == "grab" and not interact_with_same_object:
                print("[HAND] Touch exploring with both hands")
                custom_prompt = "What is the object I am touching?"
                if self.is_not_current_frame(frame_id): return
                process_single_hand('left')
                if self.is_not_current_frame(frame_id): return
                process_single_hand('right')
                # custom_prompt = "What are the objects I am touching with different hands?"
                # process_both_hands(custom_prompt, curr_left, curr_right)
                
            elif interact_with_same_object:
                print("[HAND] Both hands interacting with the same object")
                custom_prompt = "What are my hands touching?"
                if self.is_not_current_frame(frame_id): return
                process_both_hands_same_object(custom_prompt, curr_left, curr_right)
                if self.is_not_current_frame(frame_id): return
            else:
                print("[HAND] Both hands are doing different things")
                for _hand_side in hand_interaction_info['which_hands']:
                    if _hand_side == 'left' and left_change or _hand_side == 'right' and right_change:
                        if self.is_not_current_frame(frame_id): return
                        process_single_hand(_hand_side)
                        if self.is_not_current_frame(frame_id): return
                        hand_side = _hand_side
                        
            if self.is_not_current_frame(frame_id): return
        elif hand_count == 1:
            for _hand_side in hand_interaction_info['which_hands']:
                if self.is_not_current_frame(frame_id): return
                process_single_hand(_hand_side)
                if self.is_not_current_frame(frame_id): return
                hand_side = _hand_side

        if self.is_not_current_frame(frame_id): return
        self.update_moondream_buffer(packet, hand_activities, hand_interaction_info)
        if self.is_not_current_frame(frame_id): return
        end_time = time.time()
        # print("[PIPELINE] the whole pipeline takes {} seconds\n".format(end_time - start_time)*5)
        # Ensure captions exist before sending to GPT-4V
        if moondream_context:
            if self.is_not_current_frame(frame_id): return            
            threading.Thread(target=self.request_gpt4v, args=(
                self.firebaseWriteManager, self.firestoreManager, self.memoryManager,
                packet, hand_side, right_obj_image_base64, left_obj_image_base64, right_obj_mask_base64, left_obj_mask_base64,  hand_activities, 
                hand_interaction_info, self.caption_history, moondream_context
            )).start()
        else:
            print("[ERROR] No caption is generated")

    def update_moondream_buffer(self, packet, hand_activities_mediapipe, hand_interaction_info):
        temp_list = []
        
        right_change = hand_activities_mediapipe['right_change']
        left_change  = hand_activities_mediapipe['left_change']

        caption = ""
        for hand_side in hand_interaction_info['which_hands']:
            # if right_change and hand_side == 'right' or left_change and hand_side == 'left':
            if hand_side == 'right' or hand_side == 'left':
                if 'caption' in hand_interaction_info[hand_side].keys():
                    if hand_interaction_info[hand_side]['caption'] is not None:
                        if caption.strip() != hand_interaction_info[hand_side]['caption'][0]:
                            caption += hand_interaction_info[hand_side]['caption'][0] + " "
                        
                        if hand_interaction_info[hand_side]['obj_image'] is None: 
                            continue
                        
                        self.caption_history.append({
                            "caption"      : hand_interaction_info[hand_side]['caption'][0],
                            "full_image"   : packet['frame'],
                            "frame_id"     : packet['frame_id'],
                            "uuid"         : packet['uuid'],
                            "source"       : "moondream",
                            
                            "image_base64" : encode_image_to_base64(hand_interaction_info[hand_side]['obj_image']),
                            "mask_base64"  : encode_image_to_base64(hand_interaction_info[hand_side]['obj_masks']),
                            
                            "timestamp"    : packet['timestamp'],
                            "timestamp_s"  : packet['timestamp_s'],
                            
                            "hand_side"    : hand_side,
                            "left_gesture" : hand_activities_mediapipe['curr_left'],
                            "right_gesture": hand_activities_mediapipe['curr_right'],
                            
                            "hand_interaction_info": hand_interaction_info
                        })
        if not caption: return
        
        temp_list.append({
                          "caption"      : caption, 
                          "frame_id"     : packet['frame_id'], 
                          "source"       : "moondream", 
                          "uuid"         : packet['uuid'], 
                        #   "frame_base64" : packet['frame'],
                          "timestamp"    : packet['timestamp'],
                          "timestamp_s"  : packet['timestamp_s']
                        })
        
        self.ref_moondream.set(temp_list)    
        return caption

    def moondream_inference_unimanual(self, packet, image_cv2, hand_side, prompt=None):
        image_cv2 = image_cv2.astype(np.uint8)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        if image_cv2 is None:
            return None
        
        start = time.time()
        PIL_image = Image.fromarray(image_cv2)
        enc_image = model.encode_image(PIL_image)
        caption = model.answer_question(enc_image, prompt, tokenizer)

        print(f"[MOONDREAM] {hand_side}: {caption} {time.time() - start} seconds")
        return caption


    
    def moondream_inference(self, s, image_cv2, prompt=None):
        start = time.time()
        PIL_image = Image.fromarray(image_cv2)

        enc_image = model.encode_image(PIL_image)
        caption = model.answer_question(enc_image, prompt, tokenizer)

        print("[MOONDREAM] ", caption)
        print("[MOONDREAM] takes {} seconds".format(time.time() - start))
        
        return caption
    
    def get_moondream_caption_buffer(self):
        values = self.ref_moondream.get()
        if values == "start" or values == None: return "start"
        values = list(self.ref_moondream.get())
        return values
                
                
    # def listen_user_name(self,event):
        
    #     try:
    #         if self.ref_user_name_start:
    #             self.ref_user_name_start = False
    #         else:
    #             print(event.data)
    #             user_name = str(event.data)
    #             self.firestoreManager.set_name(user_name)
    #     except Exception as e:
    #         print("Errors: ", e)
    #         traceback.print_exc()  # This will print the full traceback
    #     return
    
    
    def listener_memory_image_data(self, packet):
        if self.ref_memory_image_data_start:
            self.ref_memory_image_data_start = False
            print("******************************")
            print("listening to memory logging prompts")
            print("******************************")
            return
        packet = dict(packet.data)
        self.log_memory(packet)
        
        frame_id = packet['frame_id']
        # uuid = packet['uuid']
        timestamp = packet['timestamp']
        # timestamp_s = packet['timestamp_s']
        caption = packet['caption']
        source = packet['source']
        print(f"[MEMORY LOGGING] frame_id: {frame_id}, source: {source}, timestamp: {timestamp}, caption: {caption}")

    
    def log_memory(self, packet):
        # print("packet['source']", packet['source'])
        # print("packet['caption']", packet['caption'])
        timestamp          = packet['timestamp']
        timestamp_s        = packet['timestamp_s']
        spoken_timestamp   = packet['spoken_timestamp']
        spoken_timestamp_s = packet['spoken_timestamp_s']
        frame_id           = packet['frame_id']
        
        caption            = packet['caption']
        source             = packet['source']
        _uuid              = packet['uuid'] if source == 'moondream' or source=='gpt4' else str(uuid.uuid4().hex)
        
        caption_dict = {}
        for caption_item in self.caption_history:
            if caption_item['uuid'] == _uuid and caption_item['frame_id'] == frame_id and source == caption_item['source']:
                caption_dict = caption_item
                caption_dict['uuid'] = _uuid
                
                self.firestoreManager.handscribe_update({
                    "caption"           : caption_dict['caption'],
                    "frame_id"          : caption_dict['frame_id'],
                    "source"            : caption_dict['source'],
                    "frame_base64"      : caption_dict['full_image'],
                    "timestamp"         : timestamp,
                    "timestamp_s"       : timestamp_s,  
                    "spoken_timestamp"  : spoken_timestamp,
                    "spoken_timestamp_s": spoken_timestamp_s,                         
                })
                break
        
        if not caption_dict:
            print("[ERROR] No matched caption is found in the caption history")
            self.firestoreManager.handscribe_update(packet)
            return
        
        full_image_string         = caption_dict['full_image']
        rgb_frame_string          = caption_dict['image_base64']
        mask_string               = caption_dict['mask_base64']
        hand_interaction_info     = caption_dict['hand_interaction_info']
        left_gesture              = caption_dict['left_gesture']
        right_gesture             = caption_dict['right_gesture']
        hand_side                 = caption_dict['hand_side']
        folder                    = os.path.join('memory_data', self.folder_path, str(spoken_timestamp))
        # "memory_data/" + str(spoken_timestamp)
        
        print(f"[MEMORY LOGGING] hand_side: {hand_side}, left: {left_gesture}, right: {right_gesture}")
    
        os.makedirs(folder, exist_ok=True)
        
        
        full_image = base64_to_cv2_image(full_image_string)
        full_image = cv2.rotate(full_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        full_image_path = os.path.join(folder, "full_image.png")
        cv2.imwrite(full_image_path, full_image)
        
        # Convert and save RGB image
        # print(f"[DEBUG] type of rgb_frame_string: {type(rgb_frame_string)}")
        rgb_frame = base64_to_cv2_image(rgb_frame_string)
        rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb_image_path = os.path.join(folder, "rgb_image.png")
        cv2.imwrite(rgb_image_path, rgb_frame)
        
        mask_image = base64_to_cv2_image(mask_string)
        mask_image = cv2.rotate(mask_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask_image_path = os.path.join(folder, "mask_image.png")
        cv2.imwrite(mask_image_path, mask_image)

        json_path = os.path.join(folder, "metadata.json")            
        
        # Save metadata
        packet_metadata = {
            "uuid": _uuid,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "timestamp_s": timestamp_s,
            
            "left_gesture": left_gesture,
            "right_gesture": right_gesture,
            "hand_side": hand_side,
            
            "description": caption,
            "source": source,
            # "description_embedding": encode_text(caption).tolist(),
            
            # "rgb_image_embedding": encode_image(rgb_frame).tolist(),
            # "mask_image_embedding": encode_image(mask_image).tolist(),
            # "full_image_embedding": encode_image(full_image).tolist(),
            
            "rgb_image_path": rgb_image_path,
            "mask_image_path": mask_image_path,
            "full_image_path": full_image_path,
            "json_data_path": json_path,
        }
        
        # self.memoryManager.add_data_to_table("2d_object", packet_metadata)
        with open(json_path, "w") as json_file:
            json.dump(packet_metadata, json_file, indent=4)
        
        # print(f"[MEMORY LOGGING] Descriptions: {caption}")
        # print(f"[MEMORY LOGGING] {timestamp}")
        # print(f"[MEMORY LOGGING] Saved packet data in {folder}\n\n")
                
        

    def request_gpt4v(self, firebaseWriteManager, firestoreManager, memoryManager, packet, hand_side, right_obj_image_base64, left_obj_image_base64, right_obj_mask_base64, left_obj_mask_base64, hand_activities, hand_interaction_info, caption_history, moondream_context=None):
        # hands_interact = False # TODO: change to True when memory manager is implemented
        # start_time = time.time()
        frame_id = packet['frame_id']
        
        system_role = "You are a helpful assistant that can describe the object the user is interacting with, describing the object each hand is interacting with in detail."
        user_msg = "Describe the visual details of the object each hand is interacting with."
        
        left_gesture, right_gesture               = hand_activities['curr_left'],           hand_activities['curr_right']    
        # left_index_movement, right_index_movement = hand_activities['left_index_movement'], hand_activities['right_index_movement']
        # left_change, right_change                 = hand_activities['left_change'], hand_activities['right_change']
        interact_with_same_object                 = hand_interaction_info['interact_with_same_object']
        
        if hand_side == "both":
            if interact_with_same_object:
                if left_gesture == right_gesture == 'touch_explore':
                    user_msg = f"Can you describe the spatial relationship between the two points I am touching with my index fingers? and what are the difference or similarity between the two points/objects I am touching? "
                elif left_gesture == right_gesture == 'grab':
                    user_msg = f"Can you describe the object I am holding with both hands and the parts my thumb and index finger is touching?"
                else:
                    user_msg = f"Can you describe the object I am holding and describe the part my another finger is touching?"
            else:
                user_msg = f"Can you describe the object I am holding with my left hand and the object I am holding with my right hand? and what are the difference or similarity between the two objects? What are their relationship?" 
        else:
            gesture = left_gesture if hand_side == 'left' else right_gesture
            user_msg = f"Can you describe the object I am {gesture}ing with my {hand_side} hand?"
        
        
        print("********************************************************")   
        if self.is_not_current_frame(frame_id): return  
        history = []
        # if hand_side == "right":
        #     history = memoryManager.single_search_with_img_embeddings(right_obj_mask_base64, '2d_object')
        # elif hand_side == "left":
        #     history = memoryManager.single_search_with_img_embeddings(left_obj_mask_base64, '2d_object')
        # elif hand_side == "both":
        #     if interact_with_same_object:
        #         print("debug1111111111 right_obj_mask_base64", type(right_obj_mask_base64))
        #         print("debug1111111111 left_obj_mask_base64", type(left_obj_mask_base64))
        #         history = memoryManager.single_search_with_img_embeddings(right_obj_mask_base64, '2d_object')
        #     else:  
        #         print("debug2222222222 right_obj_mask_base64", type(right_obj_mask_base64))
        #         print("debug2222222222 left_obj_mask_base64", type(left_obj_mask_base64))
        #         history = []
        #         if not right_obj_mask_base64 and left_obj_mask_base64:
        #             if right_obj_mask_base64: hand_side = "right"
        #             if left_obj_mask_base64: hand_side = "left"
        #         if right_obj_mask_base64:
        #             history += memoryManager.single_search_with_img_embeddings(right_obj_mask_base64, '2d_object') 
        #         elif left_obj_mask_base64:
        #             memoryManager.single_search_with_img_embeddings(left_obj_mask_base64, '2d_object')    
        if self.is_not_current_frame(frame_id): return
        
        
        if history:
            print("[Using historical contexts for generating descriptions.]")
            print("********************************************************")
            # history = memoryManager.single_search_with_descriptor(image_base64, '2d_object')
            payload = prepare_inputs(system_role, interact_with_same_object, right_obj_image_base64, left_obj_image_base64, user_msg, history)
        else:
            print("      Using new context from moondream descriptions.    ")
            print("********************************************************")
            payload = prepare_inputs(system_role, interact_with_same_object, right_obj_image_base64, left_obj_image_base64, user_msg, moondream_context=moondream_context)
        
        if self.is_not_current_frame(frame_id): return
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if self.is_not_current_frame(frame_id): return
        
        try:
            res = response.json()['choices'][0]['message']['content']
        except KeyError as e:
            print(f"KeyError: {e} - The response structure may be different than expected.")
            print(f"Full response: {response.json()}")
        except IndexError as e:
            print(f"IndexError: {e} - The response might have no choices or an empty list.")
            print(f"Full response: {response.json()}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            print(f"Full response: {response.text}")

        caption_list = process_response(res)
        caption_list = [".\n\n ".join(caption_list)]
        
        
        for hand_side in hand_interaction_info['which_hands']:
            # if right_change and hand_side == 'right' or left_change and hand_side == 'left':
            if hand_side == 'right' or hand_side == 'left':
                if hand_interaction_info[hand_side]['obj_image'] is None: 
                    continue
                self.caption_history.append({
                    "caption"      : caption_list[0],
                    "full_image"   : packet['frame'],
                    "frame_id"     : packet['frame_id'],
                    "uuid"         : packet['uuid'],
                    "source"       : "gpt4",
                    
                    "image_base64" : encode_image_to_base64(hand_interaction_info[hand_side]['obj_image']),
                    "mask_base64"  : encode_image_to_base64(hand_interaction_info[hand_side]['obj_masks']),
                    
                    "timestamp"    : packet['timestamp'],
                    "timestamp_s"  : packet['timestamp_s'],
                    
                    "hand_side"    : hand_side if interact_with_same_object == False else "both",
                    "left_gesture" : hand_activities['curr_left'],
                    "right_gesture": hand_activities['curr_right'],
                    
                    "hand_interaction_info": hand_interaction_info
                })
                        
        # caption_history.append({
        #                         "caption"      : caption_list[0],
        #                         "full_image"   : packet['frame'],
        #                         "frame_id"     : frame_id,
        #                         "uuid"         : packet['uuid'],
        #                         "source"       : "gpt4",

        #                         "image_base64" : image_base64,
        #                         "mask_base64"  : mask_base64,
                                
        #                         "timestamp"    : packet['timestamp'],
        #                         "timestamp_s"  : packet['timestamp_s'],
                                
        #                         "hand_side"    : hand_side,
                                
        #                         "left_gesture" : left_gesture,
        #                         "right_gesture": right_gesture,

        #                         "hand_interaction_info": hand_interaction_info
        #                     })
        
        temp_list = []
        if temp_list is not None: 
            for item in caption_list:
                temp_list.append({
                                    "caption"       : item, 
                                    "frame_id"      : frame_id,
                                    "source"        : "gpt4",
                                    "uuid"          : packet['uuid'],
                                    "frame_base64"  : packet['frame'],
                                    "timestamp"    : packet['timestamp'],
                                    "timestamp_s"  : packet['timestamp_s']
                                })
        
        if len(temp_list) == 0: 
            print("[INFO] No caption generated")
            return

        if self.is_not_current_frame(frame_id): return
        firebaseWriteManager.update_caption_buffer(temp_list)
        
        # if firebaseWriteManager: 
        #     caption_queue = firebaseWriteManager.get_caption_buffer()
        #     if caption_queue == None or caption_queue == "start" or caption_queue==['s', 't', 'a', 'r', 't']:
        #         firebaseWriteManager.update_caption_buffer(temp_list)
        #     else:
        #         if frame_id < caption_queue[-1]['frame_id']:
        #             print(f"[INFO] Frame {frame_id} is outdated than {caption_queue[-1]['frame_id']}")
        #         else:
        #             print(f"[INFO] Frame {frame_id} is processed and updated to firebase")
        #             firebaseWriteManager.update_caption_buffer(temp_list)
        #         # firestoreManager.gpt_update(rank_list)
                
        #         print(f"[INFO] GPT4v----takes {time.time()-start_time} s-----------------------\n")
        return 

    
if __name__ == '__main__':
    remote_server = RemoteServer()
