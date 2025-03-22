import torch
# from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as v2
# from tokenizer import SimpleTokenizer
# from config import get_config
# import json
# import os
from PIL import Image

# tokenizer = SimpleTokenizer()
# config = get_config()

image_transform = v2.Compose(
    [
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize([0.5], [0.5]),
    ]
)

class CLIPEmbeddingGenerator(object):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def get_image_embeddings(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.vision_model(**inputs).last_hidden_state
        
        return image_features.squeeze(0)

    def get_text_embeddings(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        
        with torch.no_grad():
            text_features = self.model.text_model(**inputs).last_hidden_state
        
        return text_features.squeeze(0)

# trinity test set
# class return_trinity(Dataset):
#     def __init__(self):
#         # get the val and test split
#         f1 = open(os.path.join(config["data_dir"]))
#         self.json_obj = json.load(f1)
#         self.clip_model = CLIPEmbeddingGenerator()
#         f1.close()
    
#     def __getitem__(self, index):
#         # get image features
#         img_pth = self.json_obj[index]['img_pth']
#         img_tensor = self.clip_model.get_image_embeddings(img_pth)

#         # get text features
#         txt_tensor = self.clip_model.get_text_embeddings(self.json_obj[index]["caps"])

#         return img_tensor, txt_tensor

#     def __len__(self):
#         return len(self.json_obj)
    
if __name__ == "__main__":
    clip_model = CLIPEmbeddingGenerator()
    img_pth = "/nfshomes/asarkar6/aditya/PRISM/generated_image.png"
    txt = "A lion is roaring on the mountain."
    img_tensor = clip_model.get_image_embeddings(img_pth)
    txt_tensor = clip_model.get_text_embeddings(txt)
    print("size image", img_tensor.shape)
    print("size text", txt_tensor.shape)





