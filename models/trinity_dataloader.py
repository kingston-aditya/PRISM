from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import os

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
        flag = 0
        try:
            img_mat = Image.open(os.path.join(self.args.dataset_name, self.temp["image"][idx])).convert("RGB")
        except Exception as e:
            flag = 1
        
        # get the image objects
        bbox_values = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info) > 0 and flag==0:
            # get the main image
            pixel_values = img_mat

            # get the prompt tokens
            with torch.no_grad():
                prompt_toks_1 = self.txt_tokenizer1(self.temp["prompt"][idx], padding="max_length", max_length=self.txt_tokenizer1.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks_2 = self.txt_tokenizer1(self.temp["prompt"][idx], padding="max_length", max_length=self.txt_tokenizer2.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks_1 = prompt_toks_1.input_ids
                prompt_toks_2 = prompt_toks_2.input_ids

            # process the bbox
            for idx, item in enumerate(bbox_info):
                x_min = int(item["xmin"])
                x_max = min(int(item["xmax"]), self.bg.size[1])
                y_min = int(item["ymin"])
                y_max = min(int(item["ymax"]), self.bg.size[0])

                if (x_max-x_min)*(y_max-y_min)>0 and y_max>=y_min and x_max>=x_min:
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

        elif len(bbox_info) == 0 or flag==1:
            pixel_values = Image.open(os.path.join(self.args.backup, "temp_img.jpg")).convert("RGB")
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks_1 = self.txt_tokenizer1("A smiling woman with a pink umbrella stands in front of a \"WELCOME TO THE LAKE\" sign, with a serene lake and green trees in the background.", padding="max_length", max_length=self.txt_tokenizer1.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks_2 = self.txt_tokenizer1("A smiling woman with a pink umbrella stands in front of a \"WELCOME TO THE LAKE\" sign, with a serene lake and green trees in the background.", padding="max_length", max_length=self.txt_tokenizer2.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks_1 = prompt_toks_1.input_ids
                prompt_toks_2 = prompt_toks_2.input_ids

            for i in range(3):
                obj_img = Image.open(os.path.join(self.args.backup, "temp_obj_"+str(i)+".jpg"))
                temp_img = self.bg
                if self.args.wanna_bg == 1:
                    try:
                        temp_img.paste(obj_img, ((y_max-y_min)//2, (x_max-x_min)//2), mask=obj_img)
                    except:
                        temp_img = obj_img
                else:
                    temp_img = obj_img
                temp_img = np.asarray(temp_img)
                bbox_values.append(Image.fromarray(temp_img))
            
        return {
            "prompt_embeds_1": prompt_toks_1,
            "prompt_embeds_2": prompt_toks_2,
            "object_prompt_embeds": bbox_values,
            "pixel_values": pixel_values,
        }

    def __len__(self):
        return len(self.temp["prompt"])
    
class SDXLInferDataset(Dataset):
    def __init__(self, temp, args, txt_tokenizer1, txt_tokenizer2):
        # load dataset
        self.temp = temp
        self.args = args
        self.txt_tokenizer1 = txt_tokenizer1
        self.txt_tokenizer2 = txt_tokenizer2
    
    def __getitem__(self, idx):
        # get the pixel values
        flag = 0
        try:
            img_mat = Image.open(os.path.join(self.args.dataset_name, self.temp["image"][idx])).convert("RGB")
        except Exception as e:
            flag = 1
        
        # get the image objects
        bbox_values = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info) > 0 and flag==0:
            # get the main image
            pixel_values = img_mat

            # get the prompt tokens
            with torch.no_grad():
                prompt_toks_1 = self.txt_tokenizer1(self.temp["prompt"][idx], padding="max_length", max_length=self.txt_tokenizer1.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks_2 = self.txt_tokenizer1(self.temp["prompt"][idx], padding="max_length", max_length=self.txt_tokenizer2.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks_1 = prompt_toks_1.input_ids
                prompt_toks_2 = prompt_toks_2.input_ids

            # process the bbox
            for idx, item in enumerate(bbox_info):
                x_min = int(item["xmin"])
                x_max = int(item["xmax"])
                y_min = int(item["ymin"])
                y_max = int(item["ymax"])

                if (x_max-x_min)*(y_max-y_min)>0 and y_max>=y_min and x_max>=x_min:
                    temp_img = np.asarray(img_mat)[y_min:y_max, x_min:x_max]
                    bbox_values.append(Image.fromarray(temp_img))

        elif len(bbox_info) == 0 or flag==1:
            pixel_values = Image.open(os.path.join(self.args.backup, "temp_img.jpg")).convert("RGB")
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks_1 = self.txt_tokenizer1("A smiling woman with a pink umbrella stands in front of a \"WELCOME TO THE LAKE\" sign, with a serene lake and green trees in the background.", padding="max_length", max_length=self.txt_tokenizer1.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks_2 = self.txt_tokenizer1("A smiling woman with a pink umbrella stands in front of a \"WELCOME TO THE LAKE\" sign, with a serene lake and green trees in the background.", padding="max_length", max_length=self.txt_tokenizer2.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks_1 = prompt_toks_1.input_ids
                prompt_toks_2 = prompt_toks_2.input_ids

            for i in range(3):
                temp_img = Image.open(os.path.join(self.args.backup, "temp_obj_"+str(i)+".jpg"))
                temp_img = np.asarray(temp_img)
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
        flag = 0
        try:
            img_mat = Image.open(os.path.join(self.args.dataset_name, self.temp["image"][idx])).convert("RGB")
        except RuntimeError as e:
            flag = 1

        # get the image objects
        bbox_values = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info)>0 and flag==0:
            # transform image and get the prompt tokens
            pixel_values = transform_img(img_mat, self.args)
            with torch.no_grad():
                prompt_toks = self.txt_tokenizer(self.temp["prompt"][idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            
            for idx, item in enumerate(bbox_info):
                x_min = int(item["xmin"])
                x_max = min(int(item["xmax"]), self.bg.size[1])
                y_min = int(item["ymin"])
                y_max = min(int(item["ymax"]), self.bg.size[0])

                if (x_max-x_min)*(y_max-y_min)>0 and y_max>=y_min and x_max>=x_min:
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
        elif len(bbox_info) == 0 or flag==1:
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks = self.txt_tokenizer("A smiling woman with a pink umbrella stands in front of a \"WELCOME TO THE LAKE\" sign, with a serene lake and green trees in the background.", max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            
            pixel_values = transform_img(Image.open(os.path.join(self.args.backup, "temp_img.jpg")).convert("RGB"), self.args)
            
            for i in range(3):
                obj_img = Image.open(os.path.join(self.args.backup, "temp_obj_"+str(i)+".jpg"))
                temp_img = self.bg
                if self.args.wanna_bg == 1:
                    try:
                        temp_img.paste(obj_img, ((y_max-y_min)//2, (x_max-x_min)//2), mask=obj_img)
                    except:
                        temp_img = obj_img
                else:
                    temp_img = obj_img
                temp_img = np.asarray(temp_img)
                bbox_values.append(Image.fromarray(temp_img))

        return {
            "prompt_embeds": prompt_toks.input_ids,
            "object_prompt_embeds": bbox_values,
            "pixel_values": pixel_values,
            "attn_mask": prompt_toks.attention_mask,
        }

    def __len__(self):
        return len(self.temp["prompt"])