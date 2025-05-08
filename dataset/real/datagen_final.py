import torch
import os
from tqdm import tqdm
import time
import json
from PIL import Image
import numpy as np
import itertools
import warnings
warnings.filterwarnings("ignore")

# get all the args
import argparse
from config import get_config
args = get_config()

def parse_args():
    parser = argparse.ArgumentParser(description="Use argparse for three params.")
    parser.add_argument('--start_len', type=int, help='STart len')
    parser.add_argument('--end_len', type=int, help='End len')
    parser.add_argument('--job_id', type=int, help='job id')
    parser.add_argument('--num_jobs', type=int, help='number of jobs thats gonna run')

    fixn_args = parser.parse_args()
    return fixn_args

import sys
sys.path.insert(1, args["repo_path"])

from utilities.run_gd import GDINO
from dataset.utils import correct_inputs, pretty_output, GD_batcher
import pdb 

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def dynamic_collate_3(batch):
    it = [item['image'] for item in batch]
    return it

def run_final_real(fixn_args):
    start_time = time.time()

    ## task 1 - load the images and filenames
    img_dataset = {"images": {}, "file_name": {}}
    
    ## task 1 - 
    ## a) get the filenames and captions
    ## b) align them if image exists else ignore it
    with open(os.path.join(args["output_metadata_folder"], "temp_imgs"+ str(fixn_args.job_id) +".json"), 'r') as f:
        img_filnames = json.load(f)
    img_dataset["file_name"] = list(img_filnames.values())

    ## load the captions and nouns
    with open(os.path.join(args["output_metadata_folder"], "temp_caps"+ str(fixn_args.job_id) +".json"), 'r') as f:
        prts = json.load(f)
    caps = list(prts["captions"].values())
    nouns = list(prts["nouns"].values())

    # load the model
    output_filname = os.path.join(args["output_metadata_folder"], "metadata"+ str(fixn_args.job_id) +".jsonl")
    gdino_obj = GDINO(args)

    # get the start batch and end batch
    print("number of batches in img dtaset", len(img_dataset["file_name"]))
    batch_start = (fixn_args.job_id-1)*(len(img_dataset["file_name"])//fixn_args.num_jobs)
    if fixn_args.job_id == fixn_args.num_jobs:
        batch_end = len(img_dataset["file_name"])
    else:
        batch_end = fixn_args.job_id*(len(img_dataset["file_name"])//fixn_args.num_jobs)

    # iterate over all the batches
    # [batch_start:batch_end]
    for k1, item in enumerate(tqdm(img_dataset["file_name"][batch_start:batch_end],desc="Processing Nouns")):
        # get the images
        img_lst = {}
        for k, img_pth in enumerate(item):
            try:
                img_lst[k] = Image.open(img_pth)
            except:
                img_lst[k] = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img_dataset_images = list(img_lst.values())

        # create batches
        temp = correct_inputs(img_dataset_images, nouns[k1], caps[k1])
        ents, imgs = GD_batcher(list(temp.values()), list(temp.keys()), 16)

        # get the GD outputs
        fin_out = {}
        for idx, k in enumerate(ents):
            lt = len(ents[idx]) # length of 1 batch of ents
            get_nouns = [item[0] for item in ents[idx]] # get the nouns of ents[idx]
            try:
                out = gdino_obj.predict(imgs[idx], get_nouns, 0.3, 0.25,)
                fin_out[idx] = out
            except:
                fin_out[idx] = [{"scores": []}]*lt

        # append the items to the file
        bbox_lst = list(itertools.chain.from_iterable(fin_out.values()))
        filname_lst = item
        nouns_caps = list(itertools.chain.from_iterable(ents))
        pretty_output(bbox_lst, filname_lst, nouns_caps, output_filname)
    
        torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total RUNTIME is {end_time - start_time}")

    del gdino_obj
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    fixn_args = parse_args()
    run_final_real(fixn_args)
