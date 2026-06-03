import torch
import base64
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

def cosine_similarity(embeddings, target_embedding, default_score=-1.0):
    """
    Compute the cosine similarity between a list of embeddings and a target embedding.
    Handles None values by assigning a default score.
    
    :param embeddings: List of numpy arrays (or None), each representing an embedding.
    :param target_embedding: A numpy array representing the target embedding.
    :param default_score: The score to assign if an embedding is None.
    :return: List of cosine similarity scores.
    """
    if target_embedding is None or not isinstance(target_embedding, np.ndarray):
        raise ValueError("Target embedding must be a valid numpy array.")

    target_embedding = target_embedding / np.linalg.norm(target_embedding)  # Normalize target

    similarities = []
    for embed in embeddings:
        if embed is None or not isinstance(embed, np.ndarray):
            similarities.append(default_score)  # Assign default score for None values
        else:
            norm_embed = np.linalg.norm(embed)
            if norm_embed == 0:
                similarities.append(default_score)  # Avoid division by zero
            else:
                similarities.append(np.dot(embed / norm_embed, target_embedding))

    return similarities

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
    return image_embedding[0].numpy()
        
def encode_text(text_any):
    text_tokens = tokenizer(text_any)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
    # Normalize the embedding
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    # print("Text Embedding Shape:", text_embedding[0])
    return text_embedding[0].numpy()
    

# texts = ["A photo of a cat"]
# print(encode_text(texts).shape)

# image = cv2.imread('/home/rueiche/worldscribe/my_images/no_hands.png')
# print(encode_image(image).shape)
