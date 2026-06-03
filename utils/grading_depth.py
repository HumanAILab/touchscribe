import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import uuid
import time
if not firebase_admin._apps:
    cred = credentials.Certificate('soundcaption-a6e7d-firebase-adminsdk-mwgfx-7e8cba13f0.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://soundcaption-a6e7d-default-rtdb.firebaseio.com/'
    })

from firebase_admin import firestore

firestore_db = firestore.client()

import gradio as gr
import json
import numpy as np
import cv2
import base64


current_participant = None
current_filename = None

def update_dropdown_address(selected_name):
    global current_participant
    global database
    current_participant = selected_name
    filenames = list(database[selected_name].keys())
    return gr.update(choices=filenames, value=filenames[0])

# def update_dropdown_sources(selected_address):
#     global current_filename
#     current_filename = selected_address
#     global description_sources
#     return gr.update(choices=description_sources, value=description_sources[0])


def load_image(selected_address):
    global database
    global current_filename
    current_filename = selected_address
    # image_filename = database[current_participant][current_filename]
    print('current_participant:', current_participant)
    print('current_filename:', current_filename)
    image  = cv2.imread(f'data/{current_participant}/{current_filename}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # print(image.shape)

    # keys_infilename = list(database[current_participant][current_filename].keys())
    caption = database[current_participant][current_filename][0]['caption']
    priority = database[current_participant][current_filename][0]['priority']
    sentences = [i for i in range(len(database[current_participant][current_filename]))]

    return image , caption , priority, gr.update(choices=sentences, value=sentences[0])

def select_sentence(selected_frame):
    global database
    global current_filename
    global current_participant
    image  = cv2.imread(f'data/{current_participant}/{current_filename}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    caption = database[current_participant][current_filename][selected_frame]['caption']
    priority = database[current_participant][current_filename][selected_frame]['priority']
    similarity_score = database[current_participant][current_filename][selected_frame]['similarity_score']
    depth_score = database[current_participant][current_filename][selected_frame]['depth_score']
    return image, caption, priority, similarity_score, depth_score

def couterclockwise(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def clockwise(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def save():
    global database

    for name in database:
        with open(f"data/priority/{name}_priority.json", "w") as outfile: 
            json.dump(database[name], outfile, indent = 4)
    print("saving...")
    print("saving complete!")
    return

def label_priority(priority):
    global database
    global current_participant
    global current_filename
    for item in database[current_participant][current_filename]:
        item['priority'] = priority
    return

if __name__ == '__main__':
    # download_data()

    database = {}
 
    import os
    import cv2
    import requests
    from utils.gpt4v import prepare_inputs, process_response, headers
    from utils.nlp_process import rank_captions
    # Path to the directory you want to search

    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # # Loop through the files in the folder

    imagenames = {}
    user_goal = {
        'paul':'Describe things on the wall',
        'vikki':'Describe posters and people in detail.',
        'diane':'Describe paintings or pictures in detail',
        'sam':'Describe any person',
        'lily':'Describe artworks or paintings.',
        'rose':"I am exploring the building. Describe things in general with more color and texture information"
    }

    user_goal_type = {
        'paul': 'specific',
        'vikki': 'specific',
        'diane': 'specific',
        'sam': 'specific',
        'lily': 'specific',
        'rose': None,
    }
    test_keys = [
                'paul',
                #  'vikki'
                
                 ]

    data = {}

    for name in list(test_keys):
        # f = open(f'data/{name}.json')
        # data = json.load(f)
        folder_path = f'data/{name}'
        data[name] = {}
        for filename in os.listdir(folder_path):
        # Check if the file has an image file extension
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                if name not in imagenames:
                    imagenames[name] = []
                imagenames[name].append(filename)
                print(filename)
    

    s_sum = 0
    s_num = 0
    d_sum = 0
    d_num = 0
    for name in test_keys:
        for filename in imagenames[name]:
            image = cv2.imread(f'data/{name}/{filename}')
            
            systemrole = "You are a helpful visual describer, who can see and describe for blind or visually imparied people. You will not mention this is an image, just describe it, and also don't mention camera issues like blurry or motion. Instead, describe things as you see as in real world but not an image. And you should describe each object with ONLY ONE sentence in maximum. Don't use 'it' to describe an object. DON'T DESCRIBE THINGS THAT ARE NOT IN THE IMAGE. Each sentence should start with a subject. Most importantly, each sentence should be at least 10 words."
            
            user_msg = "I want to understand each individual object and their visual features. And you should describe each object with ONLY ONE sentence in maximum. Don't use 'it' to describe an object. DON'T DESCRIBE THINGS THAT ARE NOT IN THE IMAGE. Each sentence should be at least 10 words."
            
            payload = prepare_inputs(systemrole, image, user_msg)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            res = response.json()['choices'][0]['message']['content']

            caption_list = process_response(res)
            temp_list = []
            if temp_list is not None: 
                for item in caption_list:
                    temp_list.append({"caption"               :item, 
                                        "similarity_score"    : None,
                                        "depth_score"         : None,
                                        "similarity_time"     : 0,
                                        "depth_time"          : 0
                                    })
            output = rank_captions(image, user_goal[name], temp_list, None, user_goal_type[name],)
            print(output)

            s_sum += output[0]['similarity_time']
            s_num +=1
            d_sum += output[0]['depth_time']
            d_num +=1
            

            data[name][filename] = output

    print("similarity time:", s_sum/s_num)
    print("depth time:", d_sum/d_num)
    for name in data: 
        with open(f"data/priority/{name}_priority.json", "w") as outfile: 
            json.dump(data[name], outfile, indent = 4)




    # database = {}
    # yes_num = 0
    # no_num = 0
    # total_num = 0
    # for name in test_keys:
    #     f = open(f'data/priority/{name}_priority.json')
    #     data = json.load(f)
    #     for filename in data:
    #         for item in data[filename]:
    #             if 'priority' not in list(item.keys()):
    #                 item['priority'] = 'NA'
            
    #         if data[filename][0]['priority'] == 'YES': yes_num+=1
    #         elif data[filename][0]['priority'] == 'NO': no_num+=1

    #     database[name] = data

    # print('YES:', yes_num, 'NO:', no_num, 'AVERAGE:', yes_num/(yes_num+no_num))

    # with gr.Blocks() as app:
    #     # Create a row layout to place elements horizontally
    #     with gr.Row():
    #         # Add an image on the left
    #         with gr.Column():
    #             image = gr.Image(value=None, width=640, height= 640,type='numpy', label="Keyframe")
    #             with gr.Row():
    #                 rotate_counterclockwise = gr.Button("-90")
    #                 rotate_clockwise = gr.Button("90")
    #             with gr.Row():
    #                 save_button = gr.Button("save to json")

    #         # Add widgets on the right
    #         with gr.Column():
    #             dropdown = gr.Dropdown(test_keys, label="Select a Name")
    #             dropdown_filename = gr.Dropdown([], label="Select a image", interactive=True)
    #             output_text = gr.Text(label="Generated Caption")
    #             similarity_score = gr.Text(label="similarity_score")
    #             depth_score      = gr.Text(label="depth_score")
    #             priority = gr.Radio(['YES', 'NO', 'NA'], label="priority", value='Not yet graded', interactive=True)
    #             sentences = gr.Radio([], label="sentences", interactive=True)
                

    #             # Define interactions
    #             dropdown.change(update_dropdown_address, inputs=[dropdown], outputs=[dropdown_filename])
    #             dropdown_filename.change(load_image, inputs=[dropdown_filename], outputs=[image, output_text, priority, sentences])
    #             sentences.change(select_sentence, inputs=[sentences], outputs=[image, output_text, priority, similarity_score, depth_score])
    #             # delays.change(label_hallucination, inputs=[frames,delays], outputs=[])
    #             priority.change(label_priority, inputs=[priority], outputs=[])
    #             rotate_counterclockwise.click(couterclockwise, inputs=[image], outputs=[image])
    #             rotate_clockwise.click(clockwise, inputs=[image], outputs=[image])
    #             save_button.click(save)

    # app.launch(share=True)
