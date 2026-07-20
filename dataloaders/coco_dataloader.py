from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import torchvision.transforms as v2
import random

import json
import os
from PIL import Image
import pdb 
from datasets import load_dataset, Dataset, IterableDataset

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/PRISM/")
from dataloaders.interleaved_dataset import InterleavedDuplicateDataset, IterableInterleavedDuplicateDataset

DATA_DIR = "/fs/cml-datasets/coco/"

class MS_COCO(InterleavedDuplicateDataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    @staticmethod
    def load_trinity_dataset(dataset_path):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        dataset = dataset.map(
            lambda example: {"prompt": [item["caption"] for item in example["annotations"]]}
        )
        dataset = dataset.map(
            lambda example: {"image_paths": [os.path.join(DATA_DIR, "images/val2017", f"{item['image_id']:012d}.jpg") for item in example["annotations"]]}
        )

        example = dataset[0]
        new_dataset = Dataset.from_dict({
            "image_path": example["image_paths"],
            "prompt": example["prompt"],
        })

        return new_dataset

class StreamingMS_COCO(IterableInterleavedDuplicateDataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    @staticmethod
    def load_your_dataset(dataset_path):
        # 1. Load the local JSON file in streaming mode
        dataset = load_dataset("json", data_files=dataset_path, split="train", streaming=True)
        
        # Inspect annotations for each sample row in the batch
        def flatten_coco_annotations(batch):
            for annotations in batch["annotations"]:
                for ann in annotations:
                    # Pull nested image_id directly from the individual annotation
                    image_id = ann.get("image_id", None)
                    caption = ann.get("caption", "")
                    
                    if image_id is None:
                        continue
                    
                    # Resolve absolute image path matching your previous script formatting
                    img_name = f"{int(image_id):012d}.jpg"
                    image_path = os.path.join(DATA_DIR, "images/val2017", img_name)
                    
                    yield {
                        "image": image_path,
                        "prompt": str(caption)
                    }

        flat_dataset = IterableDataset.from_generator(
            flatten_coco_annotations, 
            gen_kwargs={"batch": dataset}
        )
        
        return flat_dataset
    

# if __name__ == "__main__":
#     dataset = MS_COCO(dataset_path = os.path.join(DATA_DIR, "annotations", "captions_val2017.json"))
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4)
    
#     for i, batch in enumerate(dataloader):
#         pdb.set_trace()
#         break

# if __name__ == "__main__":
#     dataset = StreamingMS_COCO(
#         dataset_path=os.path.join(DATA_DIR, "annotations", "captions_val2017.json")
#     )
    
#     # Build the DataLoader with your custom static collate function
#     dataloader = DataLoader(
#         dataset,
#         batch_size=2,  # Our mock dataset will yield exactly 2 samples (1 image x 2 captions)
#         collate_fn=StreamingMS_COCO.collate_fn,
#         num_workers=0,  # Keep it 0 for clean error tracking during unit test
#     )

#     for idx, batch in enumerate(dataloader):
#         pdb.set_trace()
#         break