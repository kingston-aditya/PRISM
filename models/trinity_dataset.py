import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json

# Preprocessing the train images of dataset.
train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
train_flip = transforms.RandomHorizontalFlip(p=1.0)
train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# preprocesses images, returns 
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    # image aug
    original_sizes = []
    all_images = []
    crop_top_lefts = []
    for image in images:
        original_sizes.append((image.height, image.width))
        image = train_resize(image)
        if args.random_flip and random.random() < 0.5:
            # flip
            image = train_flip(image)
        if args.center_crop:
            y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
            x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
            image = train_crop(image)
        else:
            y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
            image = crop(image, y1, x1, h, w)
        crop_top_left = (y1, x1)
        crop_top_lefts.append(crop_top_left)
        image = train_transforms(image)
        all_images.append(image)

    return {"original_sizes": original_sizes, "crop_top_lefts": crop_top_lefts, "pixel_values": all_images}



class TrinityDataset(Dataset):
    def __init__(self, json_pth):
        # read the data
        f = open(json_pth, "r")
        json_obj = {"image":[], "prompt":[], "object":[]}
        for line in f:
            temp = json.loads(line)
            json_obj["image"].append(temp["file_name"])
            json_obj["prompt"].append(temp["prompt"])
            json_obj["object"].append(temp["object"])
        f.close()

        # process the data
        self.img_embeds = preprocess_train(json_obj)
        self.mmprompt_embeds = multimodal_encode_prompt(json_obj, ...)
        self.vae_embeds = compute_vae_encodings(self.img_embeds, vae)
    
    def __getitem__(self, idx):
        return {
            "model_input": self.vae_embeds[idx],
            "original_sizes": self.img_embeds["original_sizes"][idx],
            "crop_top_lefts": self.img_embeds["crop_top_lefts"][idx],
            "prompt_embeds": self.mmprompt_embeds["prompt_embeds"][idx],
            "pooled_prompt_embeds": self.mmprompt_embeds["pooled_prompt_embeds"][idx]
        }

    def __len__(self):
        return self.img_embeds["original_sizes"].shape[0]

if __name__ == "__main__":
    dataset = TrinityDataset("/nfshomes/asarkar6/trinity/train_data/metadata.jsonl")
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in train_loader:
        print(len(batch))
