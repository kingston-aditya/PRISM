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

import torch
import os
from tqdm import tqdm
import time
import json
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(1, args["repo_path"])
from utilities.run_gd import GDINO
from dataset.synthetic.sharegpt_dataloader import GD_batcher
from dataset.utils import correct_inputs, pretty_output
import pdb 

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def dynamic_collate_3(batch):
    it = [item['image'] for item in batch]
    return it

def run_final_real():
    start_time = time.time()

    ## task 1 - load the images and filenames
    img_dataset = {"images": {}, "file_name": {}}
    k = 0
    long_list = sorted(os.listdir(args["output_img_folder"]), key=lambda x: int(x.split(".")[0]))
    temp = [[Image.open(os.path.join(args["output_img_folder"], j)) for j in long_list[i:i + args["batch_size"]]] for i in range(0, len(long_list),  args["batch_size"])]
    for item in tqdm(temp, desc="Loading images"):
        img_dataset["images"][k] = item
        k+=1

    with open(os.path.join(args["output_metadata_folder"], "temp_imgs"+ str(args["job_id"]) +".json"), 'r') as f:
        img_filnames = json.load(f)
    img_dataset["file_name"] = img_filnames

    ## load the captions
    with open(os.path.join(args["output_metadata_folder"], "temp_caps"+ str(args["job_id"]) +".json"), 'r') as f:
        prts = json.load(f)
    caps = list(prts["captions"].values())
    nouns = list(prts["nouns"].values())

    ## task 3 - form the bounding boxes
    # create batches
    temp = correct_inputs(
        [item for sublist in list(img_dataset["images"].values()) for item in sublist],
        [item for sublist in nouns for item in sublist]
    )
    ents, imgs = GD_batcher(list(temp.keys()), list(temp.values()), 16)

    gdino_obj = GDINO(args)
    fin_out = {}; k=0
    for idx in tqdm(range(len(ents)), desc="Processing"):
        out = gdino_obj.predict(ents[idx], imgs[idx], 0.3, 0.25,)
        fin_out[k] = out
        k+=1
    del gdino_obj
    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()

    f = open(os.path.join(args["output_metadata_folder"], "metadata"+ str(args["job_id"]) +".jsonl"), "w")
    bbox_lst = [j for i in fin_out.values() for j in i]
    filname_lst = [j for i in img_dataset["file_name"].values() for j in i]
    noun_lst = [j for i in nouns for j in i]
    caps_lst = [j for i in prts["captions"].values() for j in i]
    pretty_output(bbox_lst, filname_lst, noun_lst, caps_lst, f)
    f.close()
    end_time = time.time()
    print(f"Total RUNTIME is {end_time - start_time}")

if __name__ == "__main__":
    run_final_real()

