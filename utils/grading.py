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

download_address = {
    # 'paul': ['2024-03-16 11:22:38_specific_goal',
    #          '2024-03-16 12:05:37_specific_goal',
    #          '2024-03-16 12:18:03_general_goal',
    #          '2024-03-16 12:40:46Describe more things on the wall on the wall',
    #          '2024-03-16 12:45:35Describe more things on the wall on the wall'],
    # 'vikki': [
        # '2024-03-18 13:22:12_specific_goal',
        #       '2024-03-18 13:41:03_general_goal',
        #       '2024-03-18 14:04:44_specific_goal',
            #   '2024-03-18 16:40:29Describe details of paintings on the wall and details of people'
            #   ],
    # 'diane': ['2024-03-19 13:24:52_specific_goal',
    #           '2024-03-19 13:58:54_general_goal',
    #           '2024-03-19 14:30:05_specific_goal',
    #           '2024-03-19 14:44:28Describe details on pictures signs and posters']
    # 'sam':[
    #        '2024-03-20 12:21:26_specific_goal',
    #        '2024-03-20 12:44:23_general_goal',
    #        '2024-03-20 13:19:06_specific_goal',
    #        '2024-03-20 13:42:46Describe humans in detail'],
    # 'lily':[
    #     '2024-03-23 13:33:23_specific_goal',
    #     '2024-03-23 14:05:22_general_goal',
    #     '2024-03-23 14:41:33Describe more about paintings and artworks on the wall'
    # ]
    # 'rose':[
    #     '2024-03-27 13:23:16_specific_goal',
    #     '2024-03-27 13:52:51_general_goal',
    #     '2024-03-27 14:28:37_specific_goal',
    #     '2024-03-27 14:42:42_general_goal'
    # ]
}

label_address = {
    'paul': ['2024-03-16 11:22:38_specific_goal',
            #  '2024-03-16 12:05:37_specific_goal',
             '2024-03-16 12:18:03_general_goal',
             '2024-03-16 12:40:46Describe more things on the wall on the wall',
             '2024-03-16 12:45:35Describe more things on the wall on the wall'],
    'vikki': ['2024-03-18 13:22:12_specific_goal',
              '2024-03-18 13:41:03_general_goal',
            #   '2024-03-18 14:04:44_specific_goal',
              '2024-03-18 16:40:29Describe details of paintings on the wall and details of people'
              ],
    'diane': ['2024-03-19 13:24:52_specific_goal',
              '2024-03-19 13:58:54_general_goal',
            #   '2024-03-19 14:30:05_specific_goal',
              '2024-03-19 14:44:28Describe details on pictures signs and posters'],
    'sam':[
           '2024-03-20 12:21:26_specific_goal',
           '2024-03-20 12:44:23_general_goal',
        #    '2024-03-20 13:19:06_specific_goal',
           '2024-03-20 13:42:46Describe humans in detail'],
    'lily':[
        '2024-03-23 13:33:23_specific_goal',
        '2024-03-23 14:05:22_general_goal',
        '2024-03-23 14:41:33Describe more about paintings and artworks on the wall'
    ],
    'rose':[
        '2024-03-27 13:23:16_specific_goal',
        '2024-03-27 13:52:51_general_goal',
        # '2024-03-27 14:28:37_specific_goal',
        '2024-03-27 14:42:42_general_goal'
    ]
}


description_sources = ['blip2_log', 'gpt_log', 'read_caption_log', 'moondream_log']

current_participant = None
current_address = None
current_source = 'blip2_log'


def download_data():
    names=list(download_address.keys())
    # output = {}
    # for name in names:
    # if name is None: return
    for name in names:
        # output[name]={}
        output = {}
        for addrs in download_address[name]:
            output[addrs]={}
            refs = firestore_db.collection('users').document(addrs)
            for source in description_sources:
                output[addrs][source] = []
                log = refs.collection(source).stream()
                for doc in log:
                    doc = doc.to_dict()
                    if source == "read_caption_log":
                        if doc['content']['manipulated_caption'] is None:
                            continue
                    output[addrs][source].append(doc)
                
                if source == "read_caption_log":
                    output[addrs][source] = sorted(output[addrs][source], key=lambda x: x['content']['frame_id'])

                    # print(f"{doc.id} => {doc.to_dict()}")
        
        with open(f"data/{name}.json", "w") as outfile: 
            json.dump(output, outfile, indent = 4)
            
    return

def Merge(dict1, dict2):
    return(dict2.update(dict1))

def base64_to_cv2_image(base64_string,Debug=False):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if Debug: cv2.imwrite("test.jpg", img)
    return img

def update_dropdown_address(selected_name):
    global current_participant
    current_participant = selected_name
    return gr.update(choices=label_address[selected_name], value=label_address[selected_name][0])

def update_dropdown_sources(selected_address):
    global current_address
    current_address = selected_address
    global description_sources
    return gr.update(choices=description_sources, value=description_sources[0])


def load_image(selected_address):
    global database
    global current_address
    current_address = selected_address
    read_caption_logs = database[current_participant][current_address]['read_caption_log']
    data = read_caption_logs[0]['content']
    print(data.keys())

    frames = [i for i in range(len(read_caption_logs))]
    caption = data['manipulated_caption']
    image  = base64_to_cv2_image(data['frame_base64'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # print(image.shape)
    return image , caption , gr.update(choices=frames, value=frames[0]), data['hallucination'], data['delay']

def select_frame(selected_frame):
    global database
    read_caption_logs = database[current_participant][current_address]['read_caption_log']
    data = read_caption_logs[selected_frame]['content']
    # frames = [i for i in range(len(read_caption_logs))]
    caption = data['manipulated_caption']
    hallucination = data['hallucination']
    delay = data['delay']
    image  = base64_to_cv2_image(data['frame_base64'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image, caption, hallucination, delay

def couterclockwise(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def clockwise(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def save():
    global database

    for name in database:
        with open(f"data/{name}.json", "w") as outfile: 
            json.dump(database[name], outfile, indent = 4)
    print("saving...")
    print("saving complete!")
    return

def label_hallucination(selected_frame, hallucination):
    global database
    global current_participant
    global current_address
    read_caption_logs = database[current_participant][current_address]['read_caption_log']
    data = read_caption_logs[selected_frame]['content']
    data['hallucination'] = hallucination
    return

if __name__ == '__main__':
    # download_data()

    database = {}
 
    for name in list(label_address.keys()):
        f = open(f'data/{name}.json')
        data = json.load(f)
        for addr in label_address[name]:
            read_caption_log = data[addr]['read_caption_log']

            for item in read_caption_log:
                if 'hallucination' not in list(item['content'].keys()):
                    item['content']['hallucination'] = 'NA'
                    # print('item', item)
                if 'delay' not in list(item['content'].keys()):
                    item['content']['delay'] = 'NA'

        database[name] = data
        # Merge(data, database)

    yes_num =0
    no_num =0

    yolo =0
    moon =0
    gpt4 =0

    yolo_yes =0
    moon_yes =0
    gpt4_yes =0

    yolo_no =0
    moon_no =0
    gpt4_no =0

    sentence_in_each_clip = []
    yes_in_each_clip = []
    
    for name in database:
        for addr in label_address[name]:
            read_caption_log = database[name][addr]['read_caption_log']
            sentence_num = 0
            yes_num_in_clip = 0
            for item in read_caption_log:
                if item['content']['hallucination'] == 'YES': 
                    sentence_num +=1
                    yes_num+=1
                    yes_num_in_clip +=1
                    if item['content']['source'] == 'yolo': yolo_yes+=1
                    if item['content']['source'] == 'gpt4': gpt4_yes+=1
                    if item['content']['source'] == 'moon': moon_yes+=1
                elif item['content']['hallucination'] == 'NO': 
                    sentence_num += 1
                    no_num+=1
                    if item['content']['source'] == 'yolo': yolo_no+=1
                    if item['content']['source'] == 'gpt4': gpt4_no+=1
                    if item['content']['source'] == 'moon': moon_no+=1
            sentence_in_each_clip.append(sentence_num)
            yes_in_each_clip.append(yes_num_in_clip)

    print("sentence_in_each_clip", sentence_in_each_clip)
    print("yes_in_each_clip", yes_in_each_clip)
    print(f"total number is {yes_num+no_num}")
    print(f"total number of yes is {yes_num}")
    print(yes_num/(yes_num+no_num))
    print('YOLO', yolo_yes, yolo_no, yolo_yes+ yolo_no)
    print('moon', moon_yes, moon_no, moon_yes+ moon_no)
    print('gpt4', gpt4_yes, gpt4_no, gpt4_yes+ gpt4_no)

    # print(database['paul']['2024-03-16 11:22:38_specific_goal']['read_caption_log'][0]['content'])

    with gr.Blocks() as app:
        # Create a row layout to place elements horizontally
        with gr.Row():
            # Add an image on the left
            with gr.Column():
                image = gr.Image(value=None, width=640, height= 640,type='numpy', label="Keyframe")
                with gr.Row():
                    rotate_counterclockwise = gr.Button("-90")
                    rotate_clockwise = gr.Button("90")
                with gr.Row():
                    save_button = gr.Button("save to json")

            # Add widgets on the right
            with gr.Column():
                dropdown = gr.Dropdown(list(label_address.keys()), label="Select a Name")
                dropdown_address = gr.Dropdown([], label="Select a clip", interactive=True)
                # dropdown_description_source = gr.Dropdown([], label="Select log type", interactive=True)
                # previous_button = gr.Button("Previous")
                # next_button = gr.Button("Next")
                output_text = gr.Text(label="Generated Caption")
                delays = gr.Radio(['YES', 'NO', 'NA'], label="Delay", value='Not yet graded', interactive=True)
                hallucinations = gr.Radio(['YES', 'NO', 'NA'], label="Hallucination", value='Not yet graded', interactive=True)
                frames = gr.Radio([], label="Frames", interactive=True)
                

                # Define interactions
                dropdown.change(update_dropdown_address, inputs=[dropdown], outputs=[dropdown_address])
                dropdown_address.change(load_image, inputs=[dropdown_address], outputs=[image, output_text, frames, hallucinations, delays])
                frames.change(select_frame, inputs=[frames], outputs=[image, output_text, hallucinations, delays])
                delays.change(label_hallucination, inputs=[frames,delays], outputs=[])
                hallucinations.change(label_hallucination, inputs=[frames,hallucinations], outputs=[])
                # dropdown_description_source.change(load_image, inputs=[dropdown_description_source], outputs=[image, output_text])
                rotate_counterclockwise.click(couterclockwise, inputs=[image], outputs=[image])
                rotate_clockwise.click(clockwise, inputs=[image], outputs=[image])
                save_button.click(save)

    app.launch(share=True)
