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

class CC_3M(InterleavedDuplicateDataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
    
    @staticmethod
    def load_trinity_dataset(dataset_path):
        """
        Loads the CC3M/Trinity dataset in standard map-style (non-streaming) mode.
        """
        # Load the dataset fully into memory/cache (streaming=False)
        # We specify the split directly to get a flat Dataset object rather than a DatasetDict
        dataset = load_dataset(dataset_path, split="train", streaming=False)
        
        # Standardize column names to match what your __getitem__ uses:
        # 1. Handle image pointers/paths
        if "image_path" not in dataset.column_names:
            if "image" in dataset.column_names:
                dataset = dataset.rename_column("image", "image_path")
            elif "images" in dataset.column_names:
                dataset = dataset.rename_column("images", "image_path")
                
        # 2. Handle caption/prompt strings
        if "prompt" not in dataset.column_names:
            if "caption" in dataset.column_names:
                dataset = dataset.rename_column("caption", "prompt")
            elif "text" in dataset.column_names:
                dataset = dataset.rename_column("text", "prompt")
                
        return dataset

if __name__ == "__main__":
    dataset = CC_3M(dataset_path = os.path.join())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4)
    
    for i, batch in enumerate(dataloader):
        pdb.set_trace()
        break