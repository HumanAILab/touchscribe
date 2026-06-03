from bert_score import score
import spacy
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import numpy as np
from utils.depth_estimation import get_depth
import cv2
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to_device("cuda:0")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# import CLIPSeg here
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
# import spacy model
nlp = spacy.load("en_core_web_sm")

S_THRES = 0.4
COLOR_ADJ = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "gray", "violet", "indigo", "teal", "turquoise", "lime", "olive", "maroon", "navy", "silver", "gold", "bronze", "ruby", "emerald", "amber", "sapphire", "lavender", "peach", "magenta", "scarlet", "ivory", "chocolate", "charcoal", "aquamarine", "coral", "fuchsia", "salmon", "crimson", "jade", "eggshell", "pumpkin", "hazelnut", "mustard", "cherry", "saffron", "amethyst", "buttercup", "denim", "ebony", "raspberry", "sage", "burgundy", "copper", "pearl", "periwinkle", "plum", "sienna", "tan", "tangerine", "taupe", "teal", "thistle", "umber", "vermilion", "wisteria", "zinnwaldite"]
TEXTURE_ADJ = ['Smooth', 'Rough', 'Soft', 'Hard', 'Glossy', 'Matte', 'Textured', 'Polished', 'Coarse', 'Grained', 'Fuzzy', 'Silky', 'Woolly', 'Velvety', 'Prickly', 'Gritty', 'Slick', 'Slimy', 'Lumpy', 'Bumpy', 'Crisp', 'Crumbly', 'Flaky', 'Pebbled', 'Craggy', 'Creased', 'Rigid', 'Slippery', 'Sticky', 'Tacky', 'Mushy', 'Spongy']
MATERIAL_ADJ = ["metallic", "wooden", "plastic", "fabric", "rubbery", "glassy", "paper", "ceramic", "leathery", "silk", "woolen", "cotton", "steel", "stone", "marble", "granite", "aluminium", "iron", "gold", "silver", "bronze", "copper", "titanium", "nickel", "zinc", "concrete", "clay", "terracotta", "jute", "bamboo", "ruby", "diamond", "emerald", "sapphire", "quartz", "coral", "pearl", "ivory", "jet", "amber", "garnet", "topaz", "jasper", "crystal", "obsidian", "onyx"]
SHAPE_ADJ = ['round', 'square', 'rectangular', 'circular', 'triangular', 'oval', 'cubic', 'cylindrical', 'spherical', 'hexagonal', 'polygonal', 'conical', 'pyramidal', 'tetrahedral', 'pentagonal', 'octagonal', 'oblong', 'elliptical', 'geometric', 'flat', 'curved', 'angular', 'convex', 'concave', 'prismatic']
SPATIAL_ADJ = ['above', 'below', 'beside', 'behind', 'under', 'over', 'near', 'far', 'inside', 'outside', 'adjacent', 'close', 'distant', 'within', 'without', 'beneath', 'between', 'around', 'across', 'along', 'through', 'toward', 'from', 'in', 'on', 'off', 'against', 'beside', 'onto', 'into', 'out']

def get_sentence_similarity_spacy(sentence1, sentence2):

    sentence1 = sentence1 * len(sentence2)
    output = []
    for i in range(0, len(sentence2)):
        s1 = nlp(sentence1[i])
        s2 = nlp(sentence2[i])
        similarity = s1.similarity(s2)
        output.append(similarity)
        print(f"Similarity: {similarity}")
    return output


def get_sentence_similarity(s1,s2):
    P, R, F1 = score(cands=[s1], refs=[s2]*len([s1]), lang="en", rescale_with_baseline=True)
    return F1

def concise_sentence(sentence):
    doc = nlp(sentence)
    all_adjs = COLOR_ADJ + TEXTURE_ADJ + MATERIAL_ADJ + SHAPE_ADJ
    output = []
    for token in doc:
        print(token.text, token.pos_)
        if token.text in all_adjs or token.pos_ == "ADJ" or token.pos_ == "ADV":
            print(token.text)
            continue
        sub = False
        for a in all_adjs:
            if token.text in a:
                sub = True
                break
        if not sub:
            output.append(token.text) 
    string = " ".join(output)
    string = string.split(',')[0] if ',' in string else string
    return string

def preference_check(adj_preference, sentence):
    doc = nlp(sentence)
    disabled = [i for i in adj_preference if adj_preference[i]=='disabled']
    categories = {
        "color":COLOR_ADJ,
        "material":MATERIAL_ADJ,
        "texture":TEXTURE_ADJ,
        "shape": SHAPE_ADJ
    }

    temp = []
    for d in disabled: 
        temp += categories[d]

    output = []
    for token in doc:
        if token.text in temp:
            continue
        output.append(token.text)
    
    return ' '.join(output)
    


def check_adjective(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "ADJ":
            print(token.text)
        elif token.pos_ == "ADV":
            print(token.text)
    return


def ends_with_be_verb(sentence):
    be_verbs = {"am", "is", "are", "was", "were", "be", "being", "been"}    
    words = word_tokenize(sentence)    
    tagged_words = nltk.pos_tag(words)    
    if tagged_words[-1][1].startswith('VB') and tagged_words[-1][0].lower() in be_verbs:
        return True
    return False

def get_nouns(sentence):
    doc = nlp(sentence)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    if len(noun_phrases) ==0: return ''
    else: return noun_phrases[0]


def get_depth_scores(image: np.array, prompts: list) -> list:
    thres = 200
    print("prompts", prompts)
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.unsqueeze(1)

    depth_scores = []
    for i in range(len(prompts)):
        preds_np = torch.sigmoid(preds[i][0]).cpu().numpy() * 255
        saliency = (preds_np * 255 / np.max(preds_np)).astype("uint8")
        saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]))
        depth_img = get_depth(image)
        (thresh, bw_image) = cv2.threshold(saliency, thres, 255, cv2.THRESH_BINARY)

        mask = bw_image == 255
        selected_depth_values = depth_img[mask]

        if len(prompts[i]) ==0:
            average_depth = 0
        elif selected_depth_values.size > 0:  # Check if there are any selected values
            average_depth = np.mean(selected_depth_values)
        else:
            average_depth = 0 # Or some default value, if no indices with 255 are found

        depth_scores.append(average_depth)
        # Image.fromarray(depth_img).save(f'depth{i}.jpg')
        # Image.fromarray(bw_image).save(f'clipimage{i}.jpg')
        
    return depth_scores



def get_similarity_bert(user_goal,caption_list):
    user_goal_embedding = bert_model.encode(user_goal)
    description_embeddings = bert_model.encode(caption_list)

    # Compute the cosine similarity between the user goal and each description
    similarities = util.pytorch_cos_sim(user_goal_embedding, description_embeddings)
    print("user_goal", user_goal)
    print("descriptions", len(caption_list), caption_list)
    print("similarities", similarities)
    return np.array(similarities)[0]


def rank_captions(image, user_goal, descriptions_dict, adj_preference,  user_goal_type=None):
    descriptions = [item['caption'] for item in descriptions_dict]
    nouns = [get_nouns(d) for d in descriptions]
    similarity_scores = np.zeros(len(descriptions))
    # if user_goal_type == "specific":
    #     print("user_goal", user_goal)
    #     print("descriptions", len(descriptions), descriptions)
    #     # P, R, F1 = score(cands=descriptions, refs=[user_goal]*len(descriptions), lang="en", rescale_with_baseline=True)
    #     # similarity_scores = F1.numpy()  # Convert to numpy array for easier handling
    #     similarity_scores = get_similarity_bert(user_goal, nouns)
    # else:
    #     similarity_scores = np.zeros(len(descriptions))
    depth_scores = get_depth_scores(image, nouns)
    depth_scores = np.array(depth_scores)


    for i,d in enumerate(descriptions_dict):
        d['similarity_score'] = np.float64(similarity_scores[i])
        d['depth_score'] = np.float64(depth_scores[i])

    high_similarity_list = [d for d in descriptions_dict if d['similarity_score'] >= S_THRES]
    low_similarity_list  = [d for d in descriptions_dict if d['similarity_score'] < S_THRES]
    sorted_high_similarity_list = sorted(high_similarity_list, key=lambda x: x['depth_score'], reverse=True)
    sorted_low_similarity_list  = sorted(low_similarity_list , key=lambda x: x['depth_score'], reverse=True)

    output_list = sorted_high_similarity_list+sorted_low_similarity_list
    print('----------ranking_results-----------')    
    for output_packet in output_list:
        print(f"similarity: {output_packet['similarity_score']:.2f}, Depth: {int(output_packet['depth_score'])}, Nouns: {output_packet['caption']}")
    print('------------------------------------')
    
    return output_list