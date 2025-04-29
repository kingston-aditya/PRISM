# initialize config
def get_config():
    return {
        "repo_path": "/data/home/saividyaranya/PRISM/",
        "input_data_dir": "/fsx/mrs_shlok_sai/cc12m_v2/",
        "llm_model": "Qwen/Qwen2.5-72B-Instruct",
        "mllm_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "cache_dir": "/data/home/saividyaranya/PRISM/cached_folder_real",
        "batch_size": 512,
        "dataloader_num_workers": 1,
        "is_sdxl": "False",
        "start_len": 3_000_000,
        "end_len": 4_500_000,
        "output_metadata_folder": "/data/home/saividyaranya/PRISM/cached_folder_real/metadata_folder",
        "output_img_folder": "/data/home/saividyaranya/PRISM/cached_folder_real/images/",
        "job_id": 1 
    }
args = get_config()

# import packages
from torch.utils.data import DataLoader, Subset
import torch
import os
from tqdm import tqdm
import time
import json

import warnings
warnings.filterwarnings("ignore")

from cc3m_dataloader import return_cc3_train_dataset

import pdb 

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def dynamic_collate_3(batch):
    it = [item['image'] for item in batch]
    ct = [item['caption'] for item in batch]
    return {"image": it, "caption": ct}

def run_init_real():
    ## saves image and captions
    start_time = time.time()

    ## task 1 - generate the images
    pil_dataset = return_cc3_train_dataset(args)
    indices = list(range(args["start_len"], args["end_len"])) 
    subset_data = Subset(pil_dataset, indices)

    dtel = DataLoader(
        subset_data,
        shuffle=False,
        batch_size = args["batch_size"],
        collate_fn=dynamic_collate_3
    )
    print("Number of batches", len(dtel), "Total size", len(dtel)*args["batch_size"])
    del pil_dataset
    
    k = 0; k1 = 0
    cn = {"captions":{}, "nouns":{}}
    img_dataset = {"file_name":{}, "images": {}}
    # import pdb; pdb.set_trace()
    for batch in tqdm(dtel, desc="Saving data"):
        temp = {"filname":{}, "imgs":{}}
        for img in batch["image"]:
            
            img.save( os.path.join(args["output_img_folder"], str(args["start_len"]+k1)+".png"))
            temp["filname"][k1] = os.path.join(args["output_img_folder"], str(args["start_len"]+k1)+".png")
            temp["imgs"][k1] = img
            k1 += 1
        img_dataset["file_name"][k] = list(temp["filname"].values())
        img_dataset["images"][k] = list(temp["imgs"].values())
        cn["captions"][k] = batch["caption"]
        k += 1
    end_time = time.time()

    # save dataset
    with open(os.path.join(args["output_metadata_folder"], "temp_imgs"+ str(args["job_id"]) + ".json"), 'w') as json_file:
        json.dump(img_dataset["file_name"], json_file, indent=4)
    json_file.close()

    with open(os.path.join(args["output_metadata_folder"], "temp_caps"+ str(args["job_id"]) +".json"), 'w') as json_file:
        json.dump(cn, json_file, indent=4)
    json_file.close()
        
    print(f"Total runtime of the TASK 1 is {end_time - start_time}") 

if __name__ == "__main__":
    run_init_real()