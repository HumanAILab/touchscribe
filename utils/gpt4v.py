import os
import base64
import time
from io import BytesIO
import requests
import cv2
import json
from datetime import datetime

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils.classes import SOUND_CLASSES, VISUAL_CLASSES, MANIPULATION_CLASSES, COCO_CLASSES, CUSTOM_CLASSES
from utils.nlp_process import rank_captions, ends_with_be_verb, get_nouns
from utils.increase_res_and_get_text import increase_resolution
from utils.worldscribe_utils import base64_to_cv2_image
api_key = os.environ.get("OPENAI_API_KEY", "")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

SIMILARITY_THRESHOLD = 0.6

def encode_image_to_base64(numpy_image):
    _, encoded_image = cv2.imencode('.jpg', numpy_image)
    # Convert to base64
    base64_string = base64.b64encode(encoded_image).decode('utf-8')
    return base64_string


def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def classify_user_request(message):
    # if len(CUSTOM_CLASSES) > 0:
    system_role = f"You are a powerful action classification machine, which can understand user's intent on sounds, their manipulations and visuals, and classify them into certain categories. The first one is user's goal on visuals, which contains 'general' and 'specific'. if the user's input is about their goal on visuals, such as I am looking for a sliver laptop, then you should return goal:specidic;sounds:none;manipulation:none;visuals:color;visual_objects:laptop. Another example is that when user says I want to explore the surroundings, then you should return goal:general;sounds:none;manipulation:none;visuals:none; I will introduce the other categories. First, sounds category are {SOUND_CLASSES}. Sound Manipulations are {MANIPULATION_CLASSES}. Visual Categories are {VISUAL_CLASSES}. If visual category falls into object, you should take a look on object categories {CUSTOM_CLASSES} to see which object class best match to user's need. For the return value, You need to use semicolon ; to separate different categories, and use colon to seperate category name and its value, and use comma if you have multiple value. For instance, if user say 'I want you to pause when someone speaking'. Here you need to return goal:none;sounds:speech;manipulation:pause;visuals:none;visual_objects:none to me. Because 'someone speaking' is most similar to 'speech' in the sounds category, and manipulation is similar to pause. Another example for queries on visual information is when user say 'I am curious on color and shape and dog and weather', you should return goal:none;sounds:none;manipulation:none;visuals:color,shape;visual_objects:dog to me. It is because weather does not fall into any visual category and also dog is object that in the object categories. Another example is when user say 'I want you to say less when someone knocking.' in this case you should return goal:none;sounds:knock;manipulation:talk_less;visuals:none;visual_objects:none to me. Another example is when user say 'I want to no more about blue rectangular items', in this case you should return goal:none;sounds:none;manipulation:none;visuals:color,shape;visual_objects:none to me."

    payload = {
        "model": "gpt-4",
        "messages": [
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": message}
                    ],
        "max_tokens": 800,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()['choices'][0]['message']['content']

    return res


def process_response_from_user_query(res):
    # print("hello")
    res = res.lower().strip().replace(' ', '')
    category_values = res.split(';')
    # print("category_values", category_values)
    temp = {}
    for c_v in category_values:
        c , v = c_v.split(':')
        temp[c] = v.split(',')
        if 'none' in temp[c]:
            temp[c].remove('none')
    # print(temp)
    return temp


def prepare_inputs(systemrole_default, interact_with_same_object, right_obj_image_base64, left_obj_image_base64, user_msg, history_images_texts=[], moondream_context =[], temperature=0):

    messages = [{"role": "developer", "content": [systemrole_default]}]    
    if len(moondream_context) > 0:
        moondream_context_systemrole="""You are a visual describer, you just described what is the object the user is interacting with but only with the object names. Now you should describe more visual details as the user is blind. You are looking at the object from the user's egocentric view with video streaming, so this is not an image. Describe in second-person language (using you) like you are in ..., you are looking..., the object you are holding, etc.
        """
        # if len(moondream_context) == 1:
        #     print(f"[USER_MSG] describe one object")
        #     user_msg="What does the object visually look like? Please describe in detail."
        # elif len(moondream_context) == 2:
        #     print(f"[USER_MSG] compare two objects")
        #     user_msg="Can you describe the relationship between the two objects for me? including their differences, similarities or if they can be assembled together."
        
        messages = [{"role": "system", "content": [moondream_context_systemrole]}]
        for idx, (hist_image, hist_description, which_hand, hist_date) in enumerate(moondream_context):
            history_entry =  f"Here is the previous description and the image you described to me at {hist_date}, I used my {which_hand} hand to interact with this object.\n"
            history_entry += f"History {idx+1} ({hist_date}): {hist_description}."

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": history_entry},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{hist_image}"}}
                ]
            })
        
         # Add the current query
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_msg}
            ]
        })
        
        # print(messages)
    elif len(history_images_texts) > 0:
        history_systemrole="""You are a visual describer, you need to examine the previous decription and image you provided to the user and help them reminisce the past context of the object they interacted with. 
        
        You need to answer the folling the questions in the first sentence:
        1. How long ago did the user access this object or similar object? (using rough time, such as few seconds ago, one minutes ago, two minutes ago, three minutes ago, etc.)
        2. What their hands were doing with this object? using which hand?
        3. What was the key descriptions you mentioned about this object that could help user recall? (such as color or text on the object, shape, size, that you described.)
        4. What are the differences between the current object and the previous object? (if any)
        5. Describe the object again.
        
        Here is example sentence for the same object:
        "This is [Object name] or these are [multiple same object names]. You accessed this object [how much time] ago with your [left or right] hand, you were holding it and I described [Key visual features about the object]. [Describe difference if any]. Let me describe it again [Describe the object again]."
        
        Here is example sentence for the similar object but not the same:
        "This is a new [Object name]. You accessed a similar object [how much time] ago with your [left or right] hand, I described that object [Key visual features about the object]. But it's different in terms of [Describe difference]. Let me describe this new object [Describe the new object]."

        Here is example sentence for two obviously different objects:
        "This is [Object name]. You did not access it before. [Describe the object's visual details here]."
        Make point 1, 2, 3 in one sentence.
        
        You are in the user's egocentric view with video streaming, so this is not an image. Describe in second-person language like you are in ..., you are looking..., etc. 
        """
        
        messages = [{"role": "system", "content": [history_systemrole]}]
        
        for idx, (hist_image, hist_description, hist_date, hist_seconds, left_gesture, right_gesture, hand_side) in enumerate(history_images_texts):  # Adjust number as needed
            history_entry =  f"Here is the previous description and the image you described to the user at {hist_date}.\n"
            seconds_ago = int(time.time()) - hist_seconds
            minutes = seconds_ago // 60
            seconds = seconds_ago % 60

            # Format as string
            time_string = f"{minutes} minutes" if minutes > 0 else f"{seconds} seconds"

            history_entry += f"History {idx+1} ({time_string} ago): {hist_description}. User used {hand_side} hands to interact with this object, by"
            if left_gesture != "none": history_entry += f" {left_gesture} the object with {hand_side} hand."
            if right_gesture != "none": history_entry += f" {right_gesture} the object with {hand_side} hand."

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": history_entry},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{hist_image}"}}
                ]
            })

        # user_msg="Can you describe if I accessed this object before. How long ago and what was I doing? If it is a new object, describe it in detail."
        
        print("^^^^^^^^^^right_obj_image_base64", type(right_obj_image_base64))
        print("^^^^^^^^^^left_obj_image_base64", type(left_obj_image_base64))
        # Add the current query
        if interact_with_same_object:
            content = [{"type": "text", "text": user_msg}]
            if right_obj_image_base64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{right_obj_image_base64}"}})
            messages.append({
                "role": "user",
                "content": content
            })
        else: 
            content = [{"type": "text", "text": user_msg}]
            if right_obj_image_base64:
                content.append({"type": "text", "text": "this is the object I am touching with my right hand."})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{right_obj_image_base64}"}})
            elif left_obj_image_base64:
                content.append({"type": "text", "text": "this is the object I am touching with my left hand."})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{left_obj_image_base64}"}})
            messages.append({
                "role": "user",
                "content": content
            })
    
    
    else:
        if interact_with_same_object:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{right_obj_image_base64}"}}
                ]
            })
        else: 
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{right_obj_image_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{left_obj_image_base64}"}}
                ]
            })

    
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 10000,
        "temperature": temperature
    }
    
    # print("****payload", payload)

    return payload

def request_gpt4v(firebaseWriteManager, firestoreManager, memoryManager, packet, hand_side, image_base64, mask_base64, hand_activities, hand_interaction_info, caption_history, moondream_context=None):
    hands_interact = False # TODO: change to True when memory manager is implemented
    start_time = time.time()
    frame_id = packet['frame_id']
    
    system_role = "You are a helpful assistant that can describe the object the user is interacting with, describing the object each hand is interacting with in detail."
    user_msg = "Describe the visual details of the object each hand is interacting with."
    
    left_gesture, right_gesture               = hand_activities['curr_left'],           hand_activities['curr_right']    
    left_index_movement, right_index_movement = hand_activities['left_index_movement'], hand_activities['right_index_movement']
    left_change, right_change                 = hand_activities['left_change'], hand_activities['right_change']
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
    history = memoryManager.single_search_with_img_embeddings(mask_base64, '2d_object')
    
    if history:
        print("[Using historical contexts for generating descriptions.]")
        print("********************************************************")
        # history = memoryManager.single_search_with_descriptor(image_base64, '2d_object')
        payload = prepare_inputs(system_role, image_base64, user_msg, history)
    else:
        print("      Using new context from moondream descriptions.    ")
        print("********************************************************")
        payload = prepare_inputs(system_role, image_base64, user_msg, moondream_context=moondream_context)
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    res = response.json()['choices'][0]['message']['content']

    caption_list = process_response(res)
    caption_list = [".\n\n ".join(caption_list)]
    
    
    caption_history.append({
                            "caption"      : caption_list[0],
                            "full_image"   : packet['frame'],
                            "frame_id"     : frame_id,
                            "uuid"         : packet['uuid'],
                            "source"       : "gpt4",

                            "image_base64" : image_base64,
                            "mask_base64"  : mask_base64,
                            
                            "timestamp"    : packet['timestamp'],
                            "timestamp_s"  : packet['timestamp_s'],
                            
                            "hand_side"    : hand_side,
                            
                            "left_gesture" : left_gesture,
                            "right_gesture": right_gesture,

                            "hand_interaction_info": hand_interaction_info
                        })
    
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

    if firebaseWriteManager: 
        caption_queue = firebaseWriteManager.get_caption_buffer()
        if caption_queue == None or caption_queue == "start" or caption_queue==['s', 't', 'a', 'r', 't']:
            firebaseWriteManager.update_caption_buffer(temp_list)
        else:
            if frame_id < caption_queue[-1]['frame_id']:
                print(f"[INFO] Frame {frame_id} is outdated than {caption_queue[-1]['frame_id']}")
            else:
                print(f"[INFO] Frame {frame_id} is processed and updated to firebase")
                firebaseWriteManager.update_caption_buffer(temp_list)
            # firestoreManager.gpt_update(rank_list)
            
            print(f"[INFO] GPT4v----takes {time.time()-start_time} s-----------------------\n")
    return 


def process_response(description, BLIP=False, object_preference=None):
    # print(description)
    if not description: return None
    description_list = description.strip().split('.')
    # filter out empty string and caption that has less informativeness score 5
    description_list = [item.strip() for item in description_list if len(item) !=0 
                        and cal_caption_informativeness(item,object_preference)>=0
                        ]
    
    from typing import List
    description_list: List[str]
    # print(description_list)
    output = []
    for i,item in enumerate(description_list):
        item = item.strip().lower()
        item = item.removeprefix('there appears to be')
        item = item.removeprefix("there are")
        item = item.removeprefix('there is')
        item = item.removeprefix("there's")
        item = item.removeprefix("there're")
        item = item.removeprefix("these include")
        item = item.removeprefix("a glimpse of")
        item = item.removeprefix("it displays")
        item = item.removeprefix("there seems to be a part of")
        item = item.removeprefix("there seems to be")
        item = item.removeprefix("this includes")
        item = item.replace(" also", "")
        item = item.replace("my", "your")
        item = item.replace("a partial glimpse of", "")
        item = item.replace("the image shows a person's", "")
        item = item.replace("the image shows", "")
        item = item.replace("the image features", "")
        item = item.replace("the image depicts", "")
        item = item.replace("a blurry photo of", "")
        item = item.replace("a blurry image of", "")
        item = item.replace("a blurry picture of", "")
        item = item.replace("is in the background", "")
        item = item.replace("is in the foreground", "")
        item = item.replace("is positioned in the center of the image", "")
        item = item.replace("is in the center of the image", "")
        item = item.replace("in the center of the image", "")
        item = item.replace("is the central focus of this image", "")
        item = item.replace("are the central focus of this image", "")
        item = item.replace("the central focus of this image", "")
        item = item.replace("is in the image", "")
        item = item.replace("in the image", "")
        item = item.replace("are in the background", "")
        item = item.replace("in the background", "")
        item = item.replace("are in the foreground", "")
        item = item.replace("in the foreground", "")
        item = item.replace("is depicted in this image", "")
        item = item.replace("is visible", "")
        item = item.replace("are visible", "")
        item = item.replace("it leads to", "")
        item = item.replace("you see", "")
        item = item.replace("visible", "")
        item = item.replace("partially", "")
        item = item.replace("immediate", "")
        item = item.replace("view", "")
        item = item.replace("seen", "")

        if ends_with_be_verb(item): item = get_nouns(item)
            
        output.append(item.strip())
        # print(f"[{i}] {item}")
    # print(output)
    return output



def cal_caption_informativeness(caption,object_preference=None):
    # Basic stopwords list from NLTK
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(caption)
    score = 0
    informative_words = [word for word in words if word.lower() not in stop_words]
    score += len(informative_words)
    
    # Deduct points for phrases indicating lack of information
    object_constraint = []
    if object_preference: object_constraint = [item for item in object_preference if object_preference[item] == 'false']
    negative_indicators = ['clockwise', 'ceiling', 'carpet', 'unable', 'request', 'sorry', 'orientation', 'please', 'obscures', 'rotate', 'indistinct', ' no ', ' not ' ,'blurry', 'noticeable', 'hard', 'blurred', 'not possible', 'impossible', 'obscured', 'difficult', 'blurriness', 'indiscernible', 'lack of focus', 'perhaps', 'miscellaneous', 'possibly', 'uncertain', 'unidentified', 'unclear', 'no object', 'due to the perspective']
    # negative_indicators = negative_indicators + object_constraint
    caption = caption.lower()
    for negation in negative_indicators:
        if negation in caption:
            score -= 30  # Deduct 5 points for each negative indicator
    
    # print("object_constraint", object_constraint)
    for object in object_constraint:
        if object in get_nouns(caption):
            print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            score-=30
    return score

def get_keywords_for_images(image_base64):
    systemrole = "Your role is to generate keywords and visual descriptors for the object the user is interacting with."
    user_msg = f"I am interacting with this object, please generate keywords for me. Don't consider anything in the background. Use comma to separate them. "
    payload = prepare_inputs(systemrole, image_base64, user_msg)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()['choices'][0]['message']['content']
    keywords = []
    try: 
        keywords = res.replace(' ', '').split(',')
        keywords = sorted(keywords)
        print("===keywords", keywords)
    except:
        print("errors happened in get_keywords_for_images")
        return get_keywords_for_images(image_base64)
    return keywords



def get_object_name_and_descriptors(image_base64):
    systemrole = "Your role is to identify the object the user is interacting with using their hands."
    user_msg = f"What is the object? describe with accurate adjective visual descriptors. Grammar is like 'a blue mug' or 'a metal and black cabinet' "
    payload = prepare_inputs(systemrole, image_base64, user_msg)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()['choices'][0]['message']['content']
    print("===object descriptors:", res)
    return res

if __name__ == '__main__':
    import time
    
    image = cv2.imread("test.jpg")

    # test = ['a person', 'the large painting', 'a smaller painting', 'a black trash bin', 'a metal barrier']
    # print(get_sentence_similarity_spacy(["Describe paintings or poster in detail"], test))
    # print(get_sentence_similarity_spacy(["Describe any person"], test))
    
    
    # systemrole = "You are a helpful visual describer that describe details for blind people about what they are doing"
    # SENTENCE_LENGTH_10 = "at least 10 words"
    # SENTENCE_LENGTH_20 = "at least 30 words"
    # SENTENCE_LENGTH_5 = "no longer than 5 words"
    # sentence_requirement = SENTENCE_LENGTH_20
    # user_msg = f"This is my egocentric view. Please describe what am I doing in {sentence_requirement}."
    # payload = prepare_inputs(systemrole, image, user_msg)
    # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # res = response.json()['choices'][0]['message']['content']

    
    # print(res)

    # caption_list = process_response(res)
    # print(caption_list)
    # print('----------------------------------------------')

    # temp_list = []
    # if temp_list is not None: 
    #     for item in caption_list:
    #         temp_list.append({"caption"               :item, 
    #                             "similarity_score"    : None,
    #                             "depth_score"         : None,
    #                         })
            
    # test = rank_captions(image, "Find a beige backpack on a dining chair.", temp_list, "specific")
    # print(test)