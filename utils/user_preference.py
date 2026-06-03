
import requests, os, ast
from firebase_admin import db
import traceback
import time
import ast
from utils.gpt4v import classify_user_request, process_response_from_user_query
from utils.worldscribe_utils import get_caption_from_yolo_classes, encode_image_to_base64
from utils.classes import COCO_CLASSES, OBJECT365_CLASSES,OPENIMAGE_CLASSES, CUSTOM_CLASSES_SPECIFIC, CLASSES_GOAL1, CLASSES_GOAL2, CLASSES_GOAL3, CLASSES_GOAL4, CLASSES_GOAL5
from utils.nlp_process import COLOR_ADJ, TEXTURE_ADJ, MATERIAL_ADJ, SHAPE_ADJ, SPATIAL_ADJ
# USER_GOAL = "I am looking for a silver laptop which might be on the desk and situate within many office objects."

SPECIFIC_GOAL = "I am looking for a silver laptop which might be on the desk and situate within many office objects."
GENERAL_GOAL = "I am exploring a school building freely. Describe me general information on the appliances and the building decorations."

GOAL1 = "Describe people."
GOAL2 = "Describe something on the table."
GOAL3 = "Describe clothes."
GOAL4 = "I am looking for a silver laptop."
GOAL5 = "Describe general information on nearby objects."

GOALS = [GOAL1, GOAL2, GOAL3, GOAL4, GOAL5, SPECIFIC_GOAL, GENERAL_GOAL]

AGENT_NAME = "siri"


SENTENCE_LENGTH_10 = "at least 10 words"
SENTENCE_LENGTH_20 = "at least 20 words"
SENTENCE_LENGTH_5 = "no longer than 5 words"
SENTENCE_REQUIREMENT = SENTENCE_LENGTH_10

class UserPreference:
    def __init__(self, data_processsor):

        self.user_goal = None
        self.user_goal_type = None
        self.prev_user_query = {"input": None, "timestamp":0}
        self.curr_user_query = {"input": None, "timestamp":0}
        self.agent_response = None

        self.data_processsor = data_processsor
        self.custom_classes = []

        self.ref_user_query_start = True
        self.ref_user_query = db.reference('user_query/')
        self.ref_user_query.listen(self.listener_user_query)

        self.ref_user_query_answer_start = True
        self.ref_user_query_answer = db.reference('user_query_answer/')
        self.ref_user_query_answer.listen(self.listener_user_query_answer)

    def listener_user_query(self, event):
        try:
            if self.ref_user_query_start:
                self.ref_user_query_start = False
                # print("first user_query denied")
                return
            # print('event.data', event.data)

            # s = str(event.data)
            # s = s.replace("\'", "\"")
            # dict_obj = ast.literal_eval(s)
            # print("dict_obj", dict_obj)

            dict_obj = dict(event.data)

            self.prev_user_query = self.curr_user_query

            if dict_obj['timestamp'] - self.curr_user_query['timestamp'] > 20:
                self.prev_user_query = self.curr_user_query
                self.curr_user_query['timestamp'] = dict_obj['timestamp']
                self.curr_user_query['input'] = dict_obj['input'].lower()
                print("self.prev_user_query", self.prev_user_query)
                print("self.curr_user_query", self.curr_user_query)

                if AGENT_NAME in self.curr_user_query['input'].lower():
                    # agent_response ="Hi, what can I help you?"
                    self.data_processsor.provide_reason_to_stop("invoke_agent")
                    self.data_processsor.firebaseWriteManager.send_agent_response("Hi, I am here.", 'user_can_talk')
                    # self.data_processsor.firebaseWriteManager.update_caption("Hi, I am here.")
                    print("Hi, I am here.")
                else:
                    print(f"processing user query {self.curr_user_query['input']}\n"*60)
                    self.process_user_query(dict_obj)
        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback
            # print("Event data that caused the error: ", event.data)

    def process_user_query(self, user_input):
        print("user_input: ", user_input)

        user_query = user_input['input']

        frame = self.data_processsor.current_frame
        frame_base64 = encode_image_to_base64(frame)
        packet = {
            'frame_base64': frame_base64,
            'user_query': user_query,
            'timestamp_s': time.time()
        }
        self.data_processsor.firebaseWriteManager.send_user_query_to_remote(packet)

    
        # time.sleep(3)
        # gpt_temp_response = f"I have answered your question {user_input['input']}\n"
        # self.agent_response = gpt_temp_response
            
        return
    
    def listener_user_query_answer(self, event):
        try:
            if self.ref_user_query_answer_start:
                self.ref_user_query_answer_start = False
            else:
                packet = dict(event.data)
                self.agent_response = packet['response']
        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback

        return
    


    def create_systemrole_for_interactions(self, sentence_requirement):
        base = f"You are a helpful visual describer that describe visual details for blind people about what object they are interacting with their hands."
        return base

    def create_systemrole(self, sentence_requirement):

        base = "You are a helpful visual describer, who can see and describe for blind or visually imparied people. You will not mention this is an image, just describe it, and also don't mention camera issues like blurry or motion. Instead, describe things as you see as in real world but not an image."

        verbose = [i for i in self.adj_preference if self.adj_preference[i]=='verbose']
        disabled = [i for i in self.adj_preference if self.adj_preference[i]=='disabled']
        # normal = [i for i in self.adj_preference if self.adj_preference[i]=='normal']
        if len(verbose) >0:
            base += "Blind user is very insterested in the visual features of objects represented by adjectives, such as "
            for i,t in enumerate(verbose):
                if i == len(verbose)-1:
                    base += ' and '
                if t == "color":
                    base += f"COLOR: {get_caption_from_yolo_classes(COLOR_ADJ)}, "
                elif t == "material":
                    base += f"MATERIAL: {get_caption_from_yolo_classes(MATERIAL_ADJ)}, "
                elif t == "texture":
                    base += f"TEXTURE: {get_caption_from_yolo_classes(TEXTURE_ADJ)}, "
                elif t == "shape":
                    base += f"SHAPE: {get_caption_from_yolo_classes(SHAPE_ADJ)}. "
                elif t == "spatial relationship":
                    base += f"SPATIAL RELATIONSHIPS BETWEEN OBJECTS : {get_caption_from_yolo_classes(SPATIAL_ADJ)}. "
            base += "please ensure you provide these adjective to enrich the descriptions. "
        if len(disabled) >0:
            base += "PLEASE BE AWARE YOUR WORDING! THESE ARE THE TYPES OF ADJECTIVE AND WORDS USER DON'T LIKE. DON'T INCLUDE THE FOLLOWING WORDS IN YOUR GENERATED DESCRIPTION: "
            print(disabled)
            for i,t in enumerate(disabled):
                if i == len(disabled)-1:
                    base += ' and '
                if t == "color":
                    base += f"COLOR: {get_caption_from_yolo_classes(COLOR_ADJ)}, "
                elif t == "material":
                    base += f"MATERIAL: {get_caption_from_yolo_classes(MATERIAL_ADJ)}, "
                elif t == "texture":
                    base += f"TEXTURE: {get_caption_from_yolo_classes(TEXTURE_ADJ)}, "
                elif t == "shape":
                    base += f"SHAPE: {get_caption_from_yolo_classes(SHAPE_ADJ)}, "
                elif t == "spatial relationship":
                    base += f"SPATIAL RELATIONSHIPS BETWEEN OBJECTS : {get_caption_from_yolo_classes(SPATIAL_ADJ)}. "
        

        

        base += f" And you should describe each object with ONLY ONE sentence in maximum. Don't use 'it' to describe an object. DON'T DESCRIBE THINGS THAT ARE NOT IN THE IMAGE. Most importantly, each sentence should be {sentence_requirement}."
        
        # print('base', base)

        return base
    
    def create_user_requirement_for_interaction(self, sentence_requirement):
        base = "This is my egocentric view. What object am I interactin with? Provide me with object visual details"
        return base

    def create_user_requirement(self, sentence_requirement ):
        # base = f"Please describe things based on my goal: '{self.user_goal}' and prioritize most relevant information for me when describing each object."
        base = "I want to understand each individual object and their visual features. "

        verbose = [i for i in self.adj_preference if self.adj_preference[i]=='verbose']
        disabled = [i for i in self.adj_preference if self.adj_preference[i]=='disabled']
        user_preference_adj = get_caption_from_yolo_classes(verbose)
        
        if len(verbose) >0:
            base += f"\nI am very insterested in the visual features of objects, such as {user_preference_adj}, represented by adjectives, such as "
            for i,t in enumerate(verbose):
                if i == len(verbose)-1:
                    base += ' and '
                if t == "color":
                    base += f"COLOR: {get_caption_from_yolo_classes(COLOR_ADJ)}, "
                elif t == "material":
                    base += f"MATERIAL: {get_caption_from_yolo_classes(MATERIAL_ADJ)}, "
                elif t == "texture":
                    base += f"TEXTURE: {get_caption_from_yolo_classes(TEXTURE_ADJ)}, "
                elif t == "shape":
                    base += f"SHAPE: {get_caption_from_yolo_classes(SHAPE_ADJ)}, "
                elif t == "spatial relationship":
                    base += f"SPATIAL RELATIONSHIPS BETWEEN OBJECTS : {get_caption_from_yolo_classes(SPATIAL_ADJ)}. "
            base += "Please use these adjective as much as you can."
        if len(disabled) >0:
            base += "\nPLEASE BE AWARE YOUR WORDING! THESE ARE THE TYPES OF ADJECTIVE AND WORDS I DON'T LIKE. DON'T INCLUDE THE FOLLOWING WORDS IN YOUR GENERATED DESCRIPTION: "
            print(disabled)
            for i,t in enumerate(disabled):
                if i == len(disabled)-1:
                    base += ' and '
                if t == "color":
                    base += f"COLOR: {get_caption_from_yolo_classes(COLOR_ADJ)}, "
                elif t == "material":
                    base += f"MATERIAL: {get_caption_from_yolo_classes(MATERIAL_ADJ)}, "
                elif t == "texture":
                    base += f"TEXTURE: {get_caption_from_yolo_classes(TEXTURE_ADJ)}, "
                elif t == "shape":
                    base += f"SHAPE: {get_caption_from_yolo_classes(SHAPE_ADJ)}, "
                elif t == "spatial relationship":
                    base += f"SPATIAL RELATIONSHIPS BETWEEN OBJECTS : {get_caption_from_yolo_classes(SPATIAL_ADJ)}. "
        base += f"\n And you should describe each object with ONLY ONE sentence in maximum. Don't use 'it' to describe an object. DON'T DESCRIBE THINGS THAT ARE NOT IN THE IMAGE. Each sentence should be {sentence_requirement}."
        return base

    def get_custom_classes(self, goal=None):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        if goal != None:
            message = goal
        else:
            message = "I am on university campus. I am open to know any objects in the building or on campus."

        system_role = f"You need to generate a list of at least 50 object names based on user's intent {self.user_goal}. You can first select objects based on modern datasets such as COCO: {COCO_CLASSES}, Object365: {OBJECT365_CLASSES}, Open Images V7: {OPENIMAGE_CLASSES}. Add 'person' into the list anyway. You need to sort the object list based on user's intent. The most relevant item the former. Only return a python list of object names (string) for me without any explanation."

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

        try:
            classes = ast.literal_eval(res)
            return classes
        except:
            unique_items = set(COCO_CLASSES) | set(OBJECT365_CLASSES) | set(OPENIMAGE_CLASSES)

            unique_list = list(unique_items)
            unique_list = [i.lower() for i in unique_list]

            return unique_items



if __name__ == '__main__':

    import spacy
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.neighbors import LocalOutlierFactor


    # Load the pre-trained spaCy model
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_lg")

    # Function to get the vector for a word
    

    categories = {
        "COLOR": COLOR_ADJ,
        "TEXTURE": TEXTURE_ADJ,
        "MATERIAL": MATERIAL_ADJ,
        "SHAPE": SHAPE_ADJ
    }

    # Example usage
    category_centroids = {}
    for category, adjs in categories.items():
        vectors = [nlp(adj).vector for adj in adjs if nlp(adj).has_vector]
        centroid = np.mean(vectors, axis=0)
        category_centroids[category] = centroid
        # print(centroid)

    # Function to calculate cosine similarity
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
    
    for category, adjs in categories.items():
        vectors = [nlp(adj).vector for adj in adjs if nlp(adj).has_vector]
        centroid = np.mean(vectors, axis=0)  # Averaging in the original space
        category_centroids[category] = centroid

    # Function to calculate cosine similarity between two vectors
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Function to categorize an adjective
    def categorize_adjective(adjective, threshold=0.5):
        adj_vector = nlp(adjective).vector  # Directly using the 300-dimensional vector
        similarities = {category: cosine_similarity(adj_vector, centroid) for category, centroid in category_centroids.items()}
        max_similarity = max(similarities.values())
        print(similarities)
        if max_similarity < threshold:
            return "Anomaly"  # The adjective does not closely match any category
        else:
            return max(similarities, key=similarities.get)
        
        return max(similarities, key=similarities.get)

    # Example usage
    test = ['happy','worse', 'bad','black','soft','wooden','metallic', 'hard']
    for random_adjective in test:
        category = categorize_adjective(random_adjective)
        print(f"The adjective '{random_adjective}' is classified as '{category}'.")