from torch.utils.data import Dataset

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
    
    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "prompt": self.captions[idx]
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