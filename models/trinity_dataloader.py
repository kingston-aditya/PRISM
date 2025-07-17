from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import os

def transform_obj(img, args):
    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        ]
    )
    out_img = train_transforms(img)
    return out_img

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
                    
                    try:
                        bbox_values.append(transform_obj(Image.fromarray(temp_img), self.args))
                    except:
                        print("Something wrong with the bbox", self.temp["image"][idx])
                        flag = 1

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
                bbox_values.append(transform_obj(Image.fromarray(temp_img), self.args))
            
        return {
            "prompt_embeds_1": prompt_toks_1,
            "prompt_embeds_2": prompt_toks_2,
            "object_prompt_embeds": bbox_values,
            "pixel_values": pixel_values,
            "filenames": os.path.join(self.args.dataset_name, self.temp["image"][idx])
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
        # get the image objects
        bbox_values = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info) > 0:
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks_1 = self.txt_tokenizer1(self.temp["prompt"][idx], padding="max_length", max_length=self.txt_tokenizer1.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks_2 = self.txt_tokenizer1(self.temp["prompt"][idx], padding="max_length", max_length=self.txt_tokenizer2.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks_1 = prompt_toks_1.input_ids
                prompt_toks_2 = prompt_toks_2.input_ids

            # process the bbox
            for idx, item in enumerate(bbox_info):
                img_mat = Image.open(os.path.join(self.args.valid_path_name, item["img_pth"])).convert("RGB")
                img_mat = transform_obj(img_mat, self.args)
                bbox_values.append(img_mat)

        elif len(bbox_info) == 0:
            raise Exception("Give me an object")
            
        return {
            "prompt_embeds_1": prompt_toks_1,
            "prompt_embeds_2": prompt_toks_2,
            "object_prompt_embeds": bbox_values,
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
        except Exception as e:
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
                    
                    try:
                        bbox_values.append(transform_obj(Image.fromarray(temp_img), self.args))
                    except:
                        print("Something wrong with the bbox", self.temp["image"][idx])
                        flag = 1
        
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
                bbox_values.append(transform_obj(Image.fromarray(temp_img), self.args))

        return {
            "prompt_embeds": prompt_toks.input_ids,
            "object_prompt_embeds": bbox_values,
            "pixel_values": pixel_values,
            "attn_mask": prompt_toks.attention_mask,
            "filenames": os.path.join(self.args.dataset_name, self.temp["image"][idx])
        }

    def __len__(self):
        return len(self.temp["prompt"])

class PixartInferDataset(Dataset):
    def __init__(self, temp, args, max_length, txt_tokenizer):
        # load dataset
        self.temp = temp
        self.args = args
        self.txt_tokenizer = txt_tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):        
        # get the image objects
        bbox_values = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info) > 0:
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks = self.txt_tokenizer(self.temp["prompt"][idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

            # process the bbox
            for idx, item in enumerate(bbox_info):
                img_mat = Image.open(os.path.join(self.args.valid_path_name, item["img_pth"])).convert("RGB")
                img_mat = transform_obj(img_mat, self.args)
                bbox_values.append(img_mat)

        elif len(bbox_info) == 0:
            raise Exception("Give me an object")
            
        return {
            "prompt_embeds": prompt_toks.input_ids,
            "object_prompt_embeds": bbox_values,
            "attn_mask": prompt_toks.attention_mask
        }

    def __len__(self):
        return len(self.temp["prompt"])
    
class PixartTrainDataset_pl3(Dataset):
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
        except Exception as e:
            flag = 1

        # get the image objects
        bbox_values = []
        bbox_labels = []
        bbox_labels_attnmask = []
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
                label = "An image of " + str(item["labels"])

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
                    
                    try:
                        bbox_values.append(transform_obj(Image.fromarray(temp_img), self.args))
                    except:
                        print("Something wrong with the bbox", self.temp["image"][idx])
                        flag = 1

                    temp_txt_out = self.txt_tokenizer(label, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
                    bbox_labels.append(temp_txt_out.input_ids)
                    bbox_labels_attnmask.append(temp_txt_out.attention_mask)
        
        elif len(bbox_info) == 0 or flag==1:
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks = self.txt_tokenizer("A smiling woman with a pink umbrella stands in front of a \"WELCOME TO THE LAKE\" sign, with a serene lake and green trees in the background.", max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            
            pixel_values = transform_img(Image.open(os.path.join(self.args.backup, "temp_img.jpg")).convert("RGB"), self.args)
            
            bk_labels = ["\"WELCOME TO THE LAKE\" sign", "green trees", "woman"]
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
                temp_prt = "An image of " + bk_labels[i]
                temp_txt = self.txt_tokenizer(temp_prt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

                bbox_labels.append(temp_txt.input_ids)
                bbox_labels_attnmask.append(temp_txt.attention_mask)
                bbox_values.append(transform_obj(Image.fromarray(temp_img), self.args))

        return {
            "prompt_embeds": prompt_toks.input_ids,
            "object_prompt_embeds": bbox_values,
            "object_label_embeds": bbox_labels,
            "pixel_values": pixel_values,
            "attn_mask": prompt_toks.attention_mask,
            "label_attn_mask": bbox_labels_attnmask,
            "filenames": os.path.join(self.args.dataset_name, self.temp["image"][idx])
        }

    def __len__(self):
        return len(self.temp["prompt"])
    
class PixartInferDataset_pl3(Dataset):
    def __init__(self, temp, args, max_length, txt_tokenizer):
        # load dataset
        self.temp = temp
        self.args = args
        self.txt_tokenizer = txt_tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):        
        # get the image objects
        bbox_values = []
        bbox_labels = []
        bbox_labels_attnmask = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info) > 0:
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks = self.txt_tokenizer(self.temp["prompt"][idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

            # process the bbox
            for idx, item in enumerate(bbox_info):
                # store the PIL images
                img_mat = Image.open(os.path.join(self.args.valid_path_name, item["img_pth"])).convert("RGB")
                img_mat = transform_obj(img_mat, self.args)

                # store the labels
                labs = "An image of " + str(item["labels"])
                with torch.no_grad():
                    labs_toks = self.txt_tokenizer(labs, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
                
                # appends the image matrix and label tokens
                bbox_values.append(img_mat)
                bbox_labels.append(labs_toks.input_ids)
                bbox_labels_attnmask.append(labs_toks.attention_mask)

        elif len(bbox_info) == 0:
            raise Exception("Give me an object")
            
        return {
            "prompt_embeds": prompt_toks.input_ids,
            "object_prompt_embeds": bbox_values,
            "attn_mask": prompt_toks.attention_mask,
            "label_attn_mask": bbox_labels_attnmask,
            "object_label_embeds": bbox_labels,
        }

    def __len__(self):
        return len(self.temp["prompt"])
    

class SD15TrainDataset(Dataset):
    def __init__(self, temp, args, bg, txt_tokenizer):
        # load dataset
        self.temp = temp
        self.args = args
        self.txt_tokenizer = txt_tokenizer
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
                prompt_toks = self.txt_tokenizer(self.temp["prompt"][idx], padding="max_length", max_length=self.txt_tokenizer.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks = prompt_toks.input_ids

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
                prompt_toks = self.txt_tokenizer("A smiling woman with a pink umbrella stands in front of a \"WELCOME TO THE LAKE\" sign, with a serene lake and green trees in the background.", padding="max_length", max_length=self.txt_tokenizer.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks = prompt_toks.input_ids

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
            "prompt_embeds": prompt_toks,
            "object_prompt_embeds": bbox_values,
            "pixel_values": pixel_values,
            "filenames": os.path.join(self.args.dataset_name, self.temp["image"][idx])
        }

    def __len__(self):
        return len(self.temp["prompt"])
    
class SD15InferDataset(Dataset):
    def __init__(self, temp, args, txt_tokenizer):
        # load dataset
        self.temp = temp
        self.args = args
        self.txt_tokenizer = txt_tokenizer
    
    def __getitem__(self, idx):        
        # get the image objects
        bbox_values = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info) > 0:
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks = self.txt_tokenizer(self.temp["prompt"][idx], padding="max_length", max_length=self.txt_tokenizer.model_max_length, truncation=True, return_tensors="pt")
                prompt_toks = prompt_toks.input_ids

            # process the bbox
            for idx, item in enumerate(bbox_info):
                img_mat = Image.open(os.path.join(self.args.valid_path_name, item["img_pth"])).convert("RGB")
                # img_mat = transform_obj(img_mat, self.args)
                bbox_values.append(img_mat)

        elif len(bbox_info) == 0:
            raise Exception("Give me an object.")
            
        return {
            "prompt_embeds": prompt_toks,
            "object_prompt_embeds": bbox_values,
        }

    def __len__(self):
        return len(self.temp["prompt"])

class SD15TrainDataset_pl3(Dataset):
    def __init__(self, temp, args, bg, txt_tokenizer):
        # load dataset
        self.temp = temp
        self.args = args
        self.txt_tokenizer = txt_tokenizer
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
        bbox_labels = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info)>0 and flag==0:
            # transform image and get the prompt tokens
            pixel_values = img_mat
            with torch.no_grad():
                prompt_toks = self.txt_tokenizer(self.temp["prompt"][idx], max_length=self.txt_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                prompt_toks = prompt_toks.input_ids
            
            for idx, item in enumerate(bbox_info):
                x_min = int(item["xmin"])
                x_max = min(int(item["xmax"]), self.bg.size[1])
                y_min = int(item["ymin"])
                y_max = min(int(item["ymax"]), self.bg.size[0])
                label = "An image of " + str(item["labels"])

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
                    
                    bbox_values.append(transform_obj(Image.fromarray(temp_img), self.args))
                    temp_txt_out = self.txt_tokenizer(label, max_length=self.txt_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                    bbox_labels.append(temp_txt_out.input_ids)
        
        elif len(bbox_info) == 0 or flag==1:
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks = self.txt_tokenizer("A smiling woman with a pink umbrella stands in front of a \"WELCOME TO THE LAKE\" sign, with a serene lake and green trees in the background.", max_length=self.txt_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                prompt_toks = prompt_toks.input_ids

            pixel_values = Image.open(os.path.join(self.args.backup, "temp_img.jpg")).convert("RGB")
            
            bk_labels = ["\"WELCOME TO THE LAKE\" sign", "green trees", "woman"]
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
                temp_prt = "An image of " + bk_labels[i]
                temp_txt = self.txt_tokenizer(temp_prt, max_length=self.txt_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")

                bbox_labels.append(temp_txt.input_ids)
                bbox_values.append(transform_obj(Image.fromarray(temp_img), self.args))

        return {
            "prompt_embeds": prompt_toks,
            "object_prompt_embeds": bbox_values,
            "object_label_embeds": bbox_labels,
            "pixel_values": pixel_values,
            "filenames": os.path.join(self.args.dataset_name, self.temp["image"][idx])
        }

    def __len__(self):
        return len(self.temp["prompt"])
    

class SD15InferDataset_pl3(Dataset):
    def __init__(self, temp, args, txt_tokenizer):
        # load dataset
        self.temp = temp
        self.args = args
        self.txt_tokenizer = txt_tokenizer
    
    def __getitem__(self, idx):        
        # get the image objects
        bbox_values = []
        bbox_labels = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info) > 0:
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks = self.txt_tokenizer(self.temp["prompt"][idx], max_length=self.txt_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                prompt_toks = prompt_toks.input_ids

            # process the bbox
            for idx, item in enumerate(bbox_info):
                # store the PIL images
                img_mat = Image.open(os.path.join(self.args.valid_path_name, item["img_pth"])).convert("RGB")
                img_mat = transform_obj(img_mat, self.args)

                # store the labels
                labs = "An image of " + str(item["labels"])
                with torch.no_grad():
                    labs_toks = self.txt_tokenizer(labs, max_length=self.txt_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                
                # appends the image matrix and label tokens
                bbox_values.append(img_mat)
                bbox_labels.append(labs_toks.input_ids)

        elif len(bbox_info) == 0:
            raise Exception("Give me an object")
            
        return {
            "prompt_embeds": prompt_toks,
            "object_prompt_embeds": bbox_values,
            "object_label_embeds": bbox_labels,
        }

    def __len__(self):
        return len(self.temp["prompt"])
    
class SD15_Qwen2_TrainDataset(Dataset):
    def __init__(self, temp, args, bg):
        # load dataset
        self.temp = temp
        self.args = args
        self.bg = bg
    
    def __getitem__(self, idx):
        # get the pixel values
        flag = 0
        try:
            img_mat = Image.open(os.path.join(self.args.dataset_name, self.temp["image"][idx])).convert("RGB")
        except Exception as e:
            flag = 1
        
        # train on only multimodal prompts
        if self.args.training_stage == 2:
            # get the image objects
            bbox_info = self.temp["object"][idx]

            if len(bbox_info) > 0 and flag==0:
                bbox_values = []

                # get the main image
                pixel_values = img_mat

                # get the prompt tokens
                with torch.no_grad():
                    prompt_toks = self.temp["prompt"][idx]

                # process the bbox
                for idx, item in enumerate(bbox_info):
                    x_min = int(item["xmin"])
                    x_max = min(int(item["xmax"]), self.bg.size[1])
                    y_min = int(item["ymin"])
                    y_max = min(int(item["ymax"]), self.bg.size[0])

                    if (x_max-x_min)*(y_max-y_min)>0 and y_max>y_min and x_max>x_min:
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
                        
                        if 0 in list(temp_img.shape):
                            flag = 1
                        else:
                            bbox_values.append(Image.fromarray(temp_img))

            if len(bbox_info) == 0 or flag==1:
                bbox_values = []
                pixel_values = Image.open(os.path.join(self.args.backup, "temp_img.jpg")).convert("RGB")
                # get the prompt tokens
                with torch.no_grad():
                    prompt_toks = "A smiling woman with a pink umbrella stands in front of a \"WELCOME TO THE LAKE\" sign, with a serene lake and green trees in the background."

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
        
        # train on only image reconstruction and prompt-to-text
        elif self.args.training_stage == 1:
            bbox_values = []
            if flag == 0:
                # get the prompts
                pixel_values = img_mat
                with torch.no_grad():
                    prompt_toks = self.temp["prompt"][idx]
                bbox_values.append(img_mat)
            elif flag == 1:
                # get the prompts
                pixel_values = Image.open(os.path.join(self.args.backup, "temp_img.jpg")).convert("RGB")
                prompt_toks = "A smiling woman with a pink umbrella stands in front of a \"WELCOME TO THE LAKE\" sign, with a serene lake and green trees in the background."
                bbox_values.append(pixel_values)
            else:
                raise ValueError("Flag should be either 0 or 1!!")
        else:
            raise ValueError("This training stage doesn't exist!!! Check again.")
            
        return {
            "prompts": prompt_toks,
            "object_prompt_embeds": bbox_values,
            "pixel_values": pixel_values,
            "filenames": os.path.join(self.args.dataset_name, self.temp["image"][idx])
        }

    def __len__(self):
        return len(self.temp["prompt"])


class SD15_Qwen2_InferDataset(Dataset):
    def __init__(self, temp, args, bg):
        # load dataset
        self.temp = temp
        self.args = args
        self.bg = bg
    
    def __getitem__(self, idx):        
        # get the image objects
        bbox_values = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info) > 0:
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks = self.temp["prompt"][idx]

            # process the bbox
            for idx, item in enumerate(bbox_info):
                # store the PIL images
                img_mat = Image.open(os.path.join(self.args.valid_path_name, item["img_pth"])).convert("RGB")
                img_mat = transform_obj(img_mat, self.args)
                
                # appends the image matrix and label tokens
                bbox_values.append(img_mat)

        elif len(bbox_info) == 0:
            raise Exception("Give me an object")
            
        return {
            "prompts": prompt_toks,
            "object_prompt_embeds": bbox_values,
        }

    def __len__(self):
        return len(self.temp["prompt"])
    

