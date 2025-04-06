import torch
from torch.utils.data import Dataset
import json

class ShareGPT(Dataset):
    def __init__(self, json_pth, num1, num2):
        # read json objects
        f = open(json_pth, "r")
        self.json_obj = json.load(f)
        f.close()

        # get the captions
        self.caps = []
        for i in range(len(self.json_obj)):
            prts = self.json_obj[i]["conversations"][1]['value'].split("\n\n")
            for j in prts:
                self.caps.append(j)
        
        # determine the number of captions
        if num1!=num2:
            self.caps = self.caps[num1:num2]
        
    def __getitem__(self, idx):
        return self.caps[idx]
    
    def __len__(self):
        return len(self.caps)
    
class Get_Caps(Dataset):
    def __init__(self, cn):
        # read json objects
        self.cn = cn
        
    def __getitem__(self, idx):
        return self.cn[idx]
    
    def __len__(self):
        return len(self.cn)
    
class Caps_Nouns_Filenames(Dataset):
    def __init__(self, cn, img_dataset):
        # read json objects
        self.cn = cn
        self.img_dataset = img_dataset
        
    def __getitem__(self, idx):
        return {
            "nouns": self.cn[idx],
            "images": self.img_dataset[idx]
        }
    
    def __len__(self):
        return len(self.cn)
