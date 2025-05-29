from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

def transform_img(img, args):
    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    out_img = train_transforms(img)
    return out_img

class ObjectDataset(Dataset):
    def __init__(self, img_lst):
        # read the data
        self.img_lst = img_lst
    
    def __getitem__(self, idx):
        return self.img_lst[idx]

    def __len__(self):
        return len(self.img_lst)
    
class TrinityDataset(Dataset):
    def __init__(self, json_obj):
        # load dataset
        self.images = json_obj["image"]
        self.captions = json_obj["prompt"]
        self.objects = json_obj["object"]
    
    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "prompt": self.captions[idx],
            "object": self.objects[idx],
        }

    def __len__(self):
        return len(self.images)
    
class TrinityTrainDataset(Dataset):
    def __init__(self, temp):
        # load dataset
        self.temp = temp
    
    def __getitem__(self, idx):
        return {
            "model_input": self.temp["model_input"][idx],
            "prompt_embeds": self.temp["prompt_embeds"][idx],
            "pooled_prompt_embeds": self.temp["pooled_prompt_embeds"][idx],
            "original_sizes": self.temp["original_sizes"][idx],
            "crop_top_lefts": self.temp["crop_top_lefts"][idx],
            "object_prompt_embeds": self.temp["object_prompt_embeds"][idx],
            "object_pooled_prompt_embeds": self.temp["object_pooled_prompt_embeds"][idx].reshape(1,-1)
        }

    def __len__(self):
        return len(self.temp["model_input"])

class SDXLTrainDataset(Dataset):
    def __init__(self, temp, args, bg, txt_tokenizer1, txt_tokenizer2):
        # load dataset
        self.temp = temp
        self.args = args
        self.txt_tokenizer1 = txt_tokenizer1
        self.txt_tokenizer2 = txt_tokenizer2
        self.bg = bg
    
    def __getitem__(self, idx):
        # get the pixel values
        img_mat = Image.open(self.temp["image"][idx]).convert("RGB")
        pixel_values = img_mat

        # get the prompt tokens
        with torch.no_grad():
            prompt_toks_1 = self.txt_tokenizer1(self.temp["prompt"][idx], padding="max_length", max_length=self.txt_tokenizer1.model_max_length, truncation=True, return_tensors="pt")
            prompt_toks_2 = self.txt_tokenizer1(self.temp["prompt"][idx], padding="max_length", max_length=self.txt_tokenizer2.model_max_length, truncation=True, return_tensors="pt")
            prompt_toks_1 = prompt_toks_1.input_ids
            prompt_toks_2 = prompt_toks_2.input_ids
            
        # get the image objects
        bbox_values = []
        bbox_info = self.temp["object"][idx]
        for idx, item in enumerate(bbox_info):
            x_min = int(item["xmin"])
            x_max = min(int(item["xmax"]), self.bg.size[1])
            y_min = int(item["ymin"])
            y_max = min(int(item["ymax"]), self.bg.size[0])

            if (x_max-x_min)*(y_max-y_min)>0:
                trans_y, trans_x = self.bg.size[0]//2 - (y_min+y_max)//2, self.bg.size[1]//2 - (x_min+x_max)//2
                obj_img = np.asarray(img_mat)[y_min:y_max, x_min:x_max]
                temp_img = np.asarray(self.bg)
                if self.args.wanna_bg == 1:
                    try:
                        temp_img[y_min + trans_y:y_max + trans_y, x_min+trans_x:x_max+trans_x] = obj_img
                    except:
                        temp_img = obj_img
                else:
                    temp_img = obj_img
                bbox_values.append(Image.fromarray(temp_img))
            
        return {
            "prompt_embeds_1": prompt_toks_1,
            "prompt_embeds_2": prompt_toks_2,
            "object_prompt_embeds": bbox_values,
            "pixel_values": pixel_values,
        }

    def __len__(self):
        return len(self.temp["prompt"])
    
class PixartTrainDataset(Dataset):
    def __init__(self, temp, args, bg, max_length, txt_tokenizer):
        # load dataset
        self.temp = temp
        self.args = args
        self.txt_tokenizer = txt_tokenizer
        self.max_length = max_length
        self.bg = bg
    
    def __getitem__(self, idx):
        # get the pixel values
        img_mat = Image.open(self.temp["image"][idx]).convert("RGB")
        pixel_values = transform_img(img_mat, self.args)

        # get the prompt tokens
        with torch.no_grad():
            prompt_toks = self.txt_tokenizer(self.temp["prompt"][idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        # get the image objects
        bbox_values = []
        bbox_info = self.temp["object"][idx]
        for idx, item in enumerate(bbox_info):
            x_min = int(item["xmin"])
            x_max = min(int(item["xmax"]), self.bg.size[1])
            y_min = int(item["ymin"])
            y_max = min(int(item["ymax"]), self.bg.size[0])

            if (x_max-x_min)*(y_max-y_min)>0:
                trans_y, trans_x = self.bg.size[0]//2 - (y_min+y_max)//2, self.bg.size[1]//2 - (x_min+x_max)//2
                obj_img = np.asarray(img_mat)[y_min:y_max, x_min:x_max]
                temp_img = np.asarray(self.bg)
                if self.args.wanna_bg == 1:
                    try:
                        temp_img[y_min + trans_y:y_max + trans_y, x_min+trans_x:x_max+trans_x] = obj_img
                    except:
                        temp_img = obj_img
                else:
                    temp_img = obj_img
                bbox_values.append(Image.fromarray(temp_img))
            
        return {
            "prompt_embeds": prompt_toks.input_ids,
            "object_prompt_embeds": bbox_values,
            "pixel_values": pixel_values,
            "attn_mask": prompt_toks.attention_mask,
        }

    def __len__(self):
        return len(self.temp["prompt"])