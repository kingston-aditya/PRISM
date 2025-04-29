import os
import torch

# from huggingface_hub import login
# login(token = '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')

from tqdm import tqdm
import json
import pdb 
import time

from config import get_config_2
args = get_config_2()

from sharegpt_dataloader import GD_batcher

import sys
sys.path.insert(1, args["repo_path"])
from utilities.run_diff import run_sdxl, run_flux
from utilities.run_gd import GDINO
from dataset.utils import correct_inputs, pretty_output

def run_final_syn():
    start_time = time.time()

    ## task 1 - generate the images
    with open(os.path.join(args["output_metadata_folder"], "temp_caps.json"), 'r') as f:
        prts = json.load(f)
    caps = list(prts["captions"].values())
    nouns = list(prts["nouns"].values())

    ## task 2 - forward processing
    k=0; k1=0
    if args["is_sdxl"] == "True":
        diff_obj = run_sdxl(args)
    else:
        diff_obj = run_flux(args)

    img_dataset = {"file_name":{}, "images":{}}
    for prt in tqdm(caps, desc="Processing"):
        imgs = diff_obj.forward(prt)

        a = {}; b={}
        for i in imgs:
            i.save(os.path.join(args["output_img_folder"], str(args["start_len"]+k1)+".png"))
            a[k] = os.path.join(os.path.join(args["output_img_folder"], str(args["start_len"]+k1)+".png"))
            b[k] = i
            k+=1
        img_dataset["file_name"][k1] = list(a.values())
        img_dataset["images"][k1] = list(b.values())
        k1+=1
    del diff_obj
    torch.cuda.empty_cache()
    
    # save the tensor stuff
    with open(os.path.join(args["output_metadata_folder"], "temp_images.json"), 'w') as json_file:
        json.dump(img_dataset["file_name"], json_file, indent=4)

    print("DONE WIRH SDXL")
    
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

    f = open(os.path.join(args["output_metadata_folder"], "metadata_"+str(args["job_id"])+".jsonl"), "w")
    bbox_lst = [j for i in fin_out.values() for j in i]
    filname_lst = [j for i in img_dataset["file_name"].values() for j in i]
    noun_lst = [j for i in nouns for j in i]
    caps_lst = [j for i in prts["captions"].values() for j in i]
    pretty_output(bbox_lst, filname_lst, noun_lst, caps_lst, f)
    f.close()
    end_time = time.time()
    print(f"Total RUNTIME is from run final syn {end_time - start_time}")

if __name__ == "__main__":
    run_final_syn()
