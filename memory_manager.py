from typing import List
import lancedb
import pandas as pd
import pyarrow as pa
import asyncio
import json
import requests
import time
from lancedb.embeddings import get_registry
from PIL import Image

from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
import os 

import torch
import base64
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from lancedb.pydantic import LanceModel, Vector, Dict
from lancedb.rerankers import LinearCombinationReranker
from lancedb.embeddings import get_registry
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8



api_key = os.environ.get("OPENAI_API_KEY", "")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}



model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')


def encode_image(image_any):
    image = image_any
    if isinstance(image_any, str):
        image_data = base64.b64decode(image_any)  # Decode Base64
        image = Image.open(BytesIO(image_data)) 
    elif isinstance(image_any, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image_any, cv2.COLOR_BGR2RGB))
    # Compute image embedding
    with torch.no_grad():
        image = preprocess(image).unsqueeze(0)  # Add batch dimension
        image_embedding = model.encode_image(image)
        # Normalize the embedding
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        # print("Image Embedding Shape:", image_embedding[0])
    return image_embedding[0]
        
def encode_text(text_any):
    text_tokens = tokenizer(text_any)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
    # Normalize the embedding
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    # print("Text Embedding Shape:", text_embedding[0])
    return text_embedding[0]


# func = get_registry().get("open-clip").create()


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

def encode_image_to_base64(numpy_image):
    _, encoded_image = cv2.imencode('.jpg', numpy_image)
    # Convert to base64
    base64_string = base64.b64encode(encoded_image).decode('utf-8')
    return base64_string

def prepare_inputs(systemrole, base64_image, user_msg, history_images_texts=[], temperature=0):

    # base64_image = encode_image_to_base64(image)

    messages = [{"role": "developer", "content": [systemrole]}]

    # Add historical context, pairing each historical description with its image
    if len(history_images_texts) > 0:
        history_systemrole="""You are a visual describer, you need to examine the previous decription and image you provided to the user and help them reminisce the past context of the object they interacted with. Also, please describe if the object is the same from previous one or not. If not, what are the differences? 
        Note: use the rough time instead of accurate date and time, like yesterday, last night, few days ago.
        You are in the user's egocentric view with video streaming, so this is not an image. Describe in second-person language like you are in ..., you are looking..., etc.
        """
        user_msg="Can you describe if I accessed this object before. When and what was the contexts? and is there any difference?"
        messages = [{"role": "system", "content": [history_systemrole]}]
        for idx, (hist_image, hist_description, hist_object_descriptor, hist_date) in enumerate(history_images_texts[:5]):  # Adjust number as needed
            history_entry =  f"Here is the previous description and the image you described to me at {hist_date}.\n"
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
            {"type": "text", "text": user_msg},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    })

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 10000,
        "temperature": temperature
    }
    
    print("****payload", payload)

    return payload

class HandScribeSchema(LanceModel):
    uuid: str
    timestamp: str
    frame_id: int
    timestamp_s: float
    
    
    left_gesture: str
    right_gesture: str
    hand_side: str

    description: str
    source: str
    description_embedding: Vector(1152)

    rgb_image_embedding: Vector(1152)
    mask_image_embedding: Vector(1152)
    full_image_embedding: Vector(1152)
    
    rgb_image_path: str
    json_data_path: str
    mask_image_path: str
    full_image_path: str
    
    
    class Config:
        arbitrary_types_allowed = True

    
class MemoryManager:
    def __init__(self, database_name="handscribe_db"):
        self.db = lancedb.connect(database_name)
        
        self.overwrite = True
        
        self.user_name = "rueiche" + "_"
        
        self.schema = HandScribeSchema
        
        # self.create_table(self.get_table_name('qa'), self.overwrite)
        self.create_table(self.get_table_name('2d_object'), self.overwrite)
        # self.create_table(self.get_table_name('3d_space'), self.overwrite)
        print(self.db.table_names())

    def get_table_name(self, table_name):
        if table_name in ['qa', '2d_object', '3d_space']:
            return self.user_name+table_name
        else: 
            return table_name        
    
    def get_table(self, table_name):
        table_name = self.get_table_name(table_name)
        return self.db.open_table(table_name)

    def remove_table(self, table_name):
        table_name = self.get_table_name(table_name)
        return self.db.drop_table(table_name)
    
    def create_table(self, table_name, overwrite=False):
        table_name = self.get_table_name(table_name)
        existing_tables = self.db.table_names()
        
        if table_name in existing_tables: 
            if overwrite: 
                print(f"You just create a table {table_name} for {self.user_name}")
                return self.db.create_table(table_name, schema=self.schema, mode="overwrite")
            else: 
                print(f"the table {table_name} exists. Please add data to it")
                return f"the table {table_name} exists. Please add data to it"
        else:
            return self.db.create_table(table_name, schema=self.schema)

    def add_data_to_table(self, table_name, packet_data):
        table_name = self.get_table_name(table_name)
        table = self.get_table(table_name)
        table.add([packet_data])
        return table
    
    def is_table_empty(self, table_name):
        table_name = self.get_table_name(table_name)
        table = self.get_table(table_name)
        return table.count_rows() == 0

    def get_table_columns(self, table_name):
        table_name = self.get_table_name(table_name)
        table = self.get_table(table_name)
        column_names = table.schema.names  # Extract column names
        print(f"Columns in '{table_name}': {column_names}")
        return column_names
    
    def is_column_empty(self, table_name: str, column_name: str) -> bool:
        table_name = self.get_table_name(table_name)
        table = self.get_table(table_name)
        if column_name not in table.schema.names:
            raise ValueError(f"Error: Column '{column_name}' does not exist in table '{table_name}'.")
        data_df = table.to_arrow().to_pandas()
        non_empty_count = data_df[column_name].notnull().sum()  # Count non-null values

        return non_empty_count > 0  # Returns True if data exists, False if empty

    def print_table(self, table_name: str):
        table_name = self.get_table_name(table_name)
        table = self.get_table(table_name)
        # 🔹 Convert table to Pandas DataFrame
        df = table.to_arrow().to_pandas()
        if df.empty:
            print(f"🔹 Table '{table_name}' is empty.")
        else:
            print(f"🔹 Printing table '{table_name}':")
            print(df)  # Prints the entire DataFrame
            
        
    def single_search_with_descriptor(self, image_base64, table_name):
        table_name = self.get_table_name(table_name)
        object_descriptor = get_object_name_and_descriptors(image_base64)
        object_descriptor_embedding = encode_text(object_descriptor).numpy()
        table = self.get_table(table_name)
        
        results = (
            table.search(object_descriptor_embedding, vector_column_name="object_descriptor_embedding")  # Specify vector column
            .metric("cosine")  # Use cosine similarity
            .limit(3)  # Limit results
            .to_list()  # Convert results to list
        )
        
        filtered_results = self.filter_results(results)
        
        for item in filtered_results:
            # print(f"Score (Distance): {item['_distance']:.4f}")  # Lower is better
            print(f"UUID: {item['uuid']}")
            print(f"Date: {item['date']}")
            print(f"Frame ID: {item['frame_id']}")
            print(f"Timestamp: {item['timestamp_s']}")
            print(f"Description: {item['description']}")
            print(f"Source: {item['source']}")
            print(f"RGB Image Path: {item['rgb_image_path']}")
            print("=" * 50)  # Separator for readability
        
        
        history_results = [
            (encode_image_to_base64(cv2.imread(item['rgb_image_path'])), item['description'], item['object_descriptor'], item['date'])
            for item in filtered_results
        ]
            
        return history_results
    
    def filter_results(self, search_results, thres=0.2):
        best_results_per_date = {}

        for item in search_results:
            date = item['timestamp']
            distance = item['_distance']
            
            if distance > thres:
                continue
            # If the date is not yet in the dictionary, or the new result has a better score (lower distance)
            if date not in best_results_per_date or distance < best_results_per_date[date]['_distance']:
                best_results_per_date[date] = item

        # Convert the dictionary to a sorted list (sorted by distance)
        # the latest one is the first one
        filtered_results = sorted(best_results_per_date.values(), key=lambda x: x['timestamp_s'], reverse=False)
        
        return filtered_results

            
    
    def single_search_with_img_embeddings(self, image_base64, table_name, column_name = "mask_image_embedding",  limited_search_num=3, final_num=1):
        table_name = self.get_table_name(table_name)
        
        img_embedding = encode_image(image_base64).numpy()
        table = self.get_table(table_name)
        
        results = (
            table.search(img_embedding, vector_column_name=column_name)  # Specify vector column
            .metric("cosine")  # Use cosine similarity
            .limit(limited_search_num)  # Limit results
            .to_list()  # Convert results to list
        )
        
        filtered_results = self.filter_results(results)
        print("[SEARCH RESULTS]")
        print("=" * 50)  # Separator for readability
        # for item in filtered_results:
        #     print(f"Score (Distance): {item['_distance']:.4f}")  # Lower is better
        #     print(f"UUID: {item['uuid']}")
        #     print(f"Date: {item['timestamp']}")
        #     print(f"Frame ID: {item['frame_id']}")
        #     print(f"Timestamp in s: {item['timestamp_s']}")
        #     print(f"Description: {item['description']}")
        #     print(f"Source: {item['source']}")
        #     print(f"Left gesture: {item['left_gesture']}")
        #     print(f"Right gesture: {item['right_gesture']}")
        #     print(f"which hands: {item['hand_side']}")
        #     print(f"RGB Image Path: {item['rgb_image_path']}")
            # print("=" * 50)  # Separator for readability
        
        number_of_results = 1
        history_results =[]
        if filtered_results:
            # history_results.append(
            #     (encode_image_to_base64(cv2.imread(item['rgb_image_path'])), item['description'], item['timestamp'])
            #     for item in filtered_results[:number_of_results]
            # )
            latest_results = filtered_results[-final_num:]
            for item in latest_results:
                print("=" * 50)
                print(f"[HISTORY] {item['_distance']:.4f}")
                print("[HISTORY]", item['timestamp'])
                print("[HISTORY]", item['left_gesture'])
                print("[HISTORY]", item['right_gesture'])
                print("[HISTORY]", item['hand_side'])
                print("[HISTORY]", item['description'])
                print("[HISTORY]", item['rgb_image_path'])
                print("[HISTORY]", item['mask_image_path'])
                print("=" * 50)
                history_results.append(
                    (encode_image_to_base64(cv2.imread(item['rgb_image_path'])), 
                     item['description'], 
                     item['timestamp'], 
                     item['timestamp_s'], 
                     item['left_gesture'], 
                     item['right_gesture'], 
                     item['hand_side'])
                )
        
        return history_results



if __name__ == '__main__':
    memoryManager = MemoryManager()

    # keywords = ['cup', 'cylindrical', 'drinkware', 'enamel', 'handle', 'hands', 'holding', 'metalrim', 'mug', 'white']
    query_image_path = "/home/rueiche/worldscribe/memory_data/2025-02-09_21-13-40/2/rgb_image.png"    
    image = cv2.imread(query_image_path)
    image_base64 = encode_image_to_base64(image)
    
    system_role=''
    user_msg=''
    history = memoryManager.single_search_with_descriptor(image_base64, '2d_object')
    payload = prepare_inputs(system_role, image_base64, user_msg, history)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.status_code)  # Should print 400
    print(response.text)  # Print full API response for debugging

    # query_text = get_object_name_and_descriptors(image_base64=image_base64)
    # keywords = get_keywords_for_images(image_base64=image_base64)
    # keywords_string = ", ".join(keywords)
    # print(keywords_string)
    
    # text_embedding = encode_text(query_text).numpy()
    # image_embedding = encode_image(image_base64).numpy()
    
    # table = memoryManager.get_table('rueiche_2d_object')
    
    # start_time = time.time()
    # results = (
    #     table.search(text_embedding, vector_column_name="object_descriptor_embedding")  # Specify vector column
    #     .metric("cosine")  # Use cosine similarity
    #     .limit(10)  # Limit results
    #     .to_list()  # Convert results to list
    # )
    # end_time = time.time()
    
    
    
    
    # Print results
    # for item in results:
    #     # print(f"Score (Distance): {item['_distance']:.4f}")  # Lower is better
    #     print(f"UUID: {item['uuid']}")
    #     print(f"Date: {item['date']}")
    #     print(f"Frame ID: {item['frame_id']}")
    #     print(f"Timestamp: {item['timestamp_s']}")
    #     print(f"Object Descriptor: {item['object_descriptor']}")
    #     print(f"Description: {item['description']}")
    #     print(f"Keywords: {item['keywords']}")
    #     print(f"RGB Image Path: {item['rgb_image_path']}")
    #     print("=" * 50)  # Separator for readability
        
        
    # print("spending time for search: ", end_time-start_time)
