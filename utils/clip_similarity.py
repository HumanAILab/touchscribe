import torch
from transformers import CLIPProcessor, CLIPModel
# from transformers import CLIPSegProcessor, CLIPSegModel, CLIPSegForImageSegmentation
from PIL import Image

# device = torch.device("cuda:0")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_image_similarity(text, image):

    image = Image.fromarray(image)

    # Preprocess the inputs
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

    # Forward pass through CLIP
    outputs = model(**inputs)
    text_features = outputs.text_embeds
    image_features = outputs.image_embeds

    # Normalize the features and calculate cosine similarity
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    cosine_similarity = torch.nn.functional.cosine_similarity(text_features, image_features)

    return cosine_similarity.item()