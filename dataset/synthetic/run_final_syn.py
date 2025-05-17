import os
import torch

# from huggingface_hub import login
# login(token = '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')

from tqdm import tqdm
import json
import pdb 
import time
from PIL import Image
import numpy as np

# get all the args
import argparse
from config import get_config
args = get_config()

def parse_args():
    parser = argparse.ArgumentParser(description="Use argparse for three params.")
    parser.add_argument('--start_len', type=int, help='STart len')
    parser.add_argument('--end_len', type=int, help='End len')
    parser.add_argument('--job_id', type=str, help='job id')

    fixn_args = parser.parse_args()
    return fixn_args

import sys
sys.path.insert(1, args["repo_path"])
from utilities.run_diff import run_sdxl, run_flux
from utilities.run_gd import GDINO
from dataset.utils import correct_inputs, pretty_output, GD_batcher

def run_final_syn(fixn_args):
    start_time = time.time()

    ## task 1 - generate the images
    with open(os.path.join(args["output_metadata_folder"], "temp_caps" + str(fixn_args.job_id) + ".json"), 'r') as f:
        prts = json.load(f)
    caps = prts["captions"]
    nouns = prts["nouns"]

    ## choose the diffusion model
    if args["is_sdxl"] == "True":
        diff_obj = run_sdxl(args)
    else:
        diff_obj = run_flux(args)

    # run the loop rn. We generate images.
    k=0
    img_dataset = {"file_name":[]}
    for k1, prt in enumerate(tqdm(caps, desc="Generating Images")):
        imgs = diff_obj.forward(prt)

        a = []
        for i in imgs:
            i.save(os.path.join(args["output_img_folder"], str(fixn_args.start_len+k)+".png"))
            a.append(os.path.join(os.path.join(args["output_img_folder"], str(fixn_args.start_len+k)+".png")))
            k+=1
        img_dataset["file_name"].append(a)
    del diff_obj
    torch.cuda.empty_cache()
    
    # save the tensor stuff
    with open(os.path.join(args["output_metadata_folder"], "temp_imgs" + str(fixn_args.job_id) + ".json"), 'w') as json_file:
        json.dump(img_dataset["file_name"], json_file, indent=4)

    print("DONE WIRH FLUX")
    
    # load the captions, nouns, image filenames
    k1=0
    f = open(os.path.join(args["output_metadata_folder"], "metadata"+ str(fixn_args.job_id) +".jsonl"), "w")
    gdino_obj = GDINO(args)

    # import pdb; pdb.set_trace()
    print("len of img dtaset", len(img_dataset["file_name"]))

    for k1, item in enumerate(tqdm(img_dataset["file_name"][0:len(img_dataset["file_name"])], desc="Processing Bbox")):
        
        img_lst = []
        # get the images
        for img_pth in item:
            try:
                img_lst.append(Image.open(os.path.join(args["output_img_folder"], img_pth)))
            except:
                img_lst.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
        img_dataset_images = img_lst

        ## task 3 - form the bounding boxes
        # create batches
        expanded_imgs_list, expanded_txts_list = correct_inputs(img_dataset_images, nouns[k1])
        ents, imgs = GD_batcher(expanded_imgs_list, expanded_txts_list, 4)
        # import pdb; pdb.set_trace()
        
        bbox_lst = []
        for idx in range(len(ents)):
            lt = len(ents[idx])
            try:
                out = gdino_obj.predict(imgs[idx], ents[idx], 0.3, 0.25,)
                # fin_out[k] = out
                #print("this is fine", ents[idx])
                bbox_lst.extend(out)
            except Exception as e:
                # print("exception as ", e)
                # fin_out[k] = [{"scores": []}]*lt
                #print("not fine", ents[idx])
                bbox_lst.extend([{"scores": []}]*lt)

        # append the items to the file
        # bbox_lst = [j for i in fin_out.values() for j in i]
        filname_lst = img_dataset["file_name"][k1]
        noun_lst = nouns[k1]
        caps_lst = caps[k1]
        pretty_output(bbox_lst, filname_lst, noun_lst, caps_lst, f)
        
        torch.cuda.empty_cache()

    f.close()
    end_time = time.time()
    print(f"Total RUNTIME is {end_time - start_time}")

    del gdino_obj
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    fixn_args = parse_args()
    run_final_syn(fixn_args)
