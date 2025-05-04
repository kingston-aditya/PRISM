import torch
import os
from tqdm import tqdm
import time
import json
from PIL import Image

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

    fixn_args = parser.parse_args()
    return fixn_args

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

def run_final_real(fixn_args):
    start_time = time.time()

    ## task 1 - load the images and filenames
    img_dataset = {"images": {}, "file_name": {}}
    k = 0
    long_list = sorted(os.listdir(args["output_img_folder"]), key=lambda x: int(x.split(".")[0]))[fixn_args.start_len:fixn_args.end_len]
    temp = [[Image.open(os.path.join(args["output_img_folder"], j)) for j in long_list[i:i + args["batch_size"]]] for i in range(0, len(long_list),  args["batch_size"])]
    for item in tqdm(temp, desc="Loading images"):
        img_dataset["images"][k] = item
        k+=1

    with open(os.path.join(args["output_metadata_folder"], "temp_imgs"+ str(fixn_args.job_id) +".json"), 'r') as f:
        img_filnames = json.load(f)
    img_dataset["file_name"] = img_filnames

    ## load the captions
    with open(os.path.join(args["output_metadata_folder"], "temp_caps"+ str(fixn_args.job_id) +".json"), 'r') as f:
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

    f = open(os.path.join(args["output_metadata_folder"], "metadata"+ str(fixn_args.job_id) +".jsonl"), "w")
    bbox_lst = [j for i in fin_out.values() for j in i]
    filname_lst = [j for i in img_dataset["file_name"].values() for j in i]
    noun_lst = [j for i in nouns for j in i]
    caps_lst = [j for i in prts["captions"].values() for j in i]
    pretty_output(bbox_lst, filname_lst, noun_lst, caps_lst, f)
    f.close()
    end_time = time.time()
    print(f"Total RUNTIME is {end_time - start_time}")

if __name__ == "__main__":
    fixn_args = parse_args()
    run_final_real(fixn_args)


