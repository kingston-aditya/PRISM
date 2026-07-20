import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import torch
import numpy as np
import os
from torchvision import transforms
from torchvision.transforms import functional as F
import random
from datasets import load_dataset

import os
from PIL import Image
import pdb 
import io


# Dataset curation
def get_captions_with_duplicate_objects(segments, caption, max_obj_num=3, max_size=0.5):
    if len(segments) <= max_obj_num:
        caption_object_list = segments
    else:
        raise ValueError(f"{max_obj_num} is max. Got {len(segments)}")

    return caption_object_list

    #  example:   [' Two metal chairs',
    #  <PIL.Image.Image image mode=RGB size=587x811>,
    #  ' with',
    #  ' grey cushions',
    #  <PIL.Image.Image image mode=RGB size=404x146>,
    #  ',',
    #  ' turquoise pillows',
    #  <PIL.Image.Image image mode=RGB size=332x193>,
    #  <PIL.Image.Image image mode=RGB size=499x255>,
    #  ', and',
    #  ' a table',
    #  <PIL.Image.Image image mode=RGB size=837x309>,
    #  ' with flowers on outdoor patio']

    

class RoundTo16:
    def __init__(self):
        pass

    def __call__(self, img):
        # Get original dimensions
        width, height = img.size

        # Round dimensions to nearest multiple of 16
        new_width = round(width / 16) * 16
        new_height = round(height / 16) * 16

        # Resize image to new dimensions
        return F.resize(img, (new_height, new_width), interpolation=F.InterpolationMode.BILINEAR)

class InterleavedDuplicateDataset(Dataset):
    # this dataset class doesn't process object images
    def __init__(
        self,
        dataset_path,
        size=512,
        center_crop=False,
        cache_dir=None,
    ):
        self.size = size
        self.center_crop = center_crop

        print("loading dataset")      
        # Load the dataset
        self.dataset = self.load_trinity_dataset(dataset_path)

        print("loading successful")
        print("center_crop",center_crop)

        # Define the train transformations
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.crop_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.object_image_transforms = transforms.Compose(
            [
                transforms.Resize(size//2, interpolation=transforms.InterpolationMode.BILINEAR, max_size=600),
                RoundTo16(),
            ]
        )

        # Define the transform pipeline
        self.object_image_transforms_pixel = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, max_size=600),
                RoundTo16(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        print("init successful")

    @staticmethod
    def load_load_trinity(dataset_path):
        dataset = load_dataset(dataset_path)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        # Load and process image
        try:
            target_image = Image.open(self.dataset[index]["image_path"]) if self.dataset[index]["image_path"] is not None else self.dataset[index]["images"]
            target_image = target_image.convert('RGB')

            if not target_image.mode == "RGB":
                target_image = target_image.convert("RGB")
        except Exception as e:
            new_index = random.randint(0,len(self.dataset)-1)
            return self.__getitem__(new_index)
        
        example["target_image"] = self.image_transforms(target_image)

        segments = [self.object_image_transforms(target_image)]*1

        example["obj_name_image"] = get_captions_with_duplicate_objects(segments, str(self.dataset[index]["prompt"]))

        example["prompt"] = str(self.dataset[index]["prompt"])

        return example

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]

        obj_name_images = [example["obj_name_image"] for example in examples]

        targets = [example["target_image"] for example in examples]

        targets_pixel_values = torch.stack(targets) if targets[0] is not None else None
        targets_pixel_values = targets_pixel_values.to(memory_format=torch.contiguous_format).float() if targets_pixel_values is not None else None

        batch = {"prompt": prompts, "target_image": targets_pixel_values, "obj_name_image":obj_name_images}
        return batch

class IterableInterleavedDuplicateDataset(IterableDataset):
    def __init__(self, dataset_path, size=512, center_crop=False,):
        super().__init__()
        self.dataset = self.load_your_dataset(dataset_path)
        
        # Define the train transformations
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.object_image_transforms = transforms.Compose(
            [
                transforms.Resize(size//2, interpolation=transforms.InterpolationMode.BILINEAR, max_size=600),
                RoundTo16(),
                transforms.ToTensor(),
            ]
        )

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading (num_workers=0)
            sharded_dataset = self.dataset
        else:
            # Multi-process data loading: Assign a unique subset of shards to this worker
            sharded_dataset = self.dataset.shard(
                num_shards=worker_info.num_workers, 
                index=worker_info.id
            )

        dataset_iterator = iter(sharded_dataset)
        while True:
            try:
                # 1. Fetch next sequential raw sample
                sample = next(dataset_iterator)
                
                # Extract properties based on your CC3M schema
                image_source = sample.get("image", None)
                prompt = sample.get("prompt", sample.get("caption", ""))
                
                if image_source is None:
                    continue

                # 2. Open and Convert Image safely
                if isinstance(image_source, Image.Image):
                    target_image = image_source
                elif isinstance(image_source, str):
                    target_image = Image.open(image_source)
                else:
                    target_image = Image.open(io.BytesIO(image_source))

                # Force PIL to decode the data to verify it isn't truncated/corrupt
                target_image.draft(target_image.mode, target_image.size) 
                target_image = target_image.convert('RGB')
                
                # 3. Assemble the processed payload
                example = {}
                example["target_image"] = self.image_transforms(target_image)
                
                # Handle your segment crops
                segments = [self.object_image_transforms(target_image)] * 1
                example["obj_name_image"] = get_captions_with_duplicate_objects(segments, str(prompt))
                example["prompt"] = str(prompt)
                
                # Yield the successful sample up to the DataLoader
                yield example

            except StopIteration:
                # End of stream reached cleanly
                break
            except Exception as e:
                # If ANY image parsing or transformer error happens,
                # we silently skip it and let the loop pull the next sequence
                print(f"Skipping corrupt sample. Error: {e}")
                continue
    
    @staticmethod
    def load_your_dataset(dataset_path):
        dataset = load_dataset(dataset_path, split="train", streaming=True)
        return dataset
    
    @staticmethod
    def collate_fn(batch):
        target_images = torch.stack([item["target_image"] for item in batch], dim=0)
        prompts = [item["prompt"] for item in batch]
        obj_name_images = [item["obj_name_image"] for item in batch]
        
        return {
            "target_image": target_images,
            "obj_name_image": obj_name_images,
            "prompt": prompts
        }