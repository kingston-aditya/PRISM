import torch
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore")

from sharegpt_dataloader import ShareGPT, Get_Caps, Caps_Nouns_Filenames
from utils import correct_inputs, pretty_output, dynamic_collate, dynamic_collate_1

from config import get_config
config1 = get_config()

import sys
sys.path.insert(1, config1["repo_path"])
from utilities import run_gd, run_llm
from utilities.run_diff import run_sdxl, run_flux

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a data generation script.")
    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        required=True,
        help="LLM to be used.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/model_weights/",
        help="Cache directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size",
    )
    parser.add_argument(
        "--total_length_yes",
        type=str,
        default="False",
        help="Train all samples or fixed number of samples",
    )
    parser.add_argument(
        "--total_length_no",
        type=int,
        default=20e3,
        help="Total number of samples",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=12,
        help="Number of workers",
    )
    parser.add_argument(
        "--is_sdxl",
        type=str,
        default="True",
        help="To use SDXL or not",
    )
    parser.add_argument(
        "--output_img_folder",
        type=str,
        default="/nfshomes/asarkar6/trinity/finale_data/images",
        help="Where to save dataset",
    )
    parser.add_argument(
        "--output_metadata_folder",
        type=str,
        default="/nfshomes/asarkar6/trinity/finale_data/",
        help="Where to save dataset",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

class generate_syn_data(object):
    def __init__(self, json_pth, args):
        if args.total_length_yes == "True":
            self.caps_dataset = ShareGPT(json_pth, -1) 
        else:
            self.caps_dataset = ShareGPT(json_pth, args.total_length_no)
        self.args = args
        
    def forward(self):
        # task 1 - get the nouns and captions
        cn = {"captions": {}, "nouns": {}}
        dtel = DataLoader(
            self.caps_dataset,
            shuffle=True,
            batch_size = self.args.batch_size,
            collate_fn=dynamic_collate,
            num_workers=self.args.dataloader_num_workers,
        )
        print("Number of batches", len(dtel), "Total size", len(dtel)*self.args.batch_size)
        self.llm_obj = run_llm.run_qwen(self.args)
        k = 0
        for batch in tqdm(dtel, desc="Processing"):
            caps = self.llm_obj.get_summary(batch)
            nouns = self.llm_obj.get_nouns(caps)
            # saving for further processing
            cn["captions"][k] = caps
            cn["nouns"][k] = nouns
            # update counter
            k+=1
            del caps, nouns
        del self.llm_obj, dtel, self.caps_dataset
        torch.cuda.empty_cache()

        # flushing intermediate output
        with open(os.path.join(self.args.output_metadata_folder, "temp_caps.json"), 'w') as json_file:
            json.dump(cn["captions"], json_file, indent=4)

        # task 2 - get the images
        if self.args.is_sdxl == "True":
            self.diff_obj = run_sdxl(self.args)
        else:
            self.diff_obj = run_flux(self.args)
        
        get_caps = Get_Caps([i for j in cn["captions"].values() for i in j])
        dtel = DataLoader(
            get_caps,
            shuffle=False,
            batch_size = self.args.batch_size,
            collate_fn=dynamic_collate,
            num_workers=self.args.dataloader_num_workers,
        )

        k = 0
        img_dataset = {"file_name":{}, "images": {}}
        for batch in dtel:
            img_gen = self.diff_obj.forward(batch)
            for img in img_gen:
                k+=1
                img.save(os.path.join(self.args.output_img_folder, str(k)+".png"))
                img_dataset["file_name"][k] = os.path.join(self.args.output_img_folder, str(k)+".png")
                img_dataset["images"][k] = img
            del img_gen
        del self.diff_obj, dtel, get_caps
        torch.cuda.empty_cache()
        
        # task 3 - get the objects
        self.GD = run_gd.GDINO(args)
        get_caps_nouns_filenames = Caps_Nouns_Filenames([j for i in cn["nouns"].values() for j in i], [k for k in img_dataset["images"].values()])
        dtel = DataLoader(
            get_caps_nouns_filenames,
            shuffle=False,
            batch_size = self.args.batch_size,
            collate_fn=dynamic_collate_1,
            num_workers=self.args.dataloader_num_workers,
        )
        fin_out = {}
        k = 0
        for batch in tqdm(dtel, desc="processing"):
            temp = correct_inputs(batch["images"], batch["nouns"])
            out = self.GD.predict(list(temp.values()), list(temp.keys()), 0.3, 0.25,)
            fin_out[k] = out
            k+=1
        del dtel, self.GD
        torch.cuda.empty_cache()
        
        f = open(os.path.join(self.args.output_metadata_folder, "metadata.jsonl"), "w")
        bbox_lst = [j for i in fin_out.values() for j in i]
        filname_lst = [j for j in img_dataset["file_name"].values()]
        noun_lst = [j for i in cn["nouns"].values() for j in i]
        caps_lst = [j for i in cn["captions"].values() for j in i]
        pretty_output(bbox_lst, filname_lst, noun_lst, caps_lst, f)
        f.close()

if __name__ == "__main__":
    # for pipeline 5
    args = parse_args()
    json_pth = config1["data_path"]
    generate_syn_data(json_pth, args).forward()


