import torch
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore")

from sharegpt_dataloader import ShareGPT, Caps_Nouns_Filenames
from utils import correct_inputs, pretty_output, dynamic_collate, dynamic_collate_2

from config import get_config
config1 = get_config()

import sys
sys.path.insert(1, config1["repo_path"])
from utilities import run_gd, run_llm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a data generation script.")
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
        "--is_sdxl",
        type=str,
        default="True",
        help="To use SDXL or not",
    )
    parser.add_argument(
        "--start_len",
        type=int,
        default=0,
        help="Starting length",
    )
    parser.add_argument(
        "--end_len",
        type=int,
        default=1024,
        help="Ending length",
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

class generate_part_data(object):
    def __init__(self, json_pth, args):
        if args.total_length_yes == "True":
            self.caps_dataset = ShareGPT(json_pth, -1, -1) 
        else:
            self.caps_dataset = ShareGPT(json_pth, args.start_len, args.end_len)
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
            # summartizes the caption
            caps = self.llm_obj.get_summary(batch)
            # get the nouns from summarized captipon
            nouns = self.llm_obj.get_nouns(caps)

            # saving for further processing
            cn["captions"][k] = caps
            cn["nouns"][k] = nouns
            k+=1

            # flushing intermediate output
            with open(os.path.join(self.args.output_metadata_folder, "temp_caps.json"), 'w') as json_file:
                json.dump(cn, json_file, indent=4)
            json_file.close()

            del caps, nouns
        del self.llm_obj, dtel, self.caps_dataset
        torch.cuda.empty_cache()

        # task 2 - get the filenames
        filnames = sorted(os.path.listdir(self.args.output_image_folder), key=lambda x: int(x.split('.')[0]))
        
        # task 3 - get the objects
        self.GD = run_gd.GDINO(args)
        get_caps_nouns_filenames = Caps_Nouns_Filenames([j for i in cn["nouns"].values() for j in i], filnames)
        dtel = DataLoader(
            get_caps_nouns_filenames,
            shuffle=False,
            batch_size = self.args.batch_size,
            collate_fn=dynamic_collate_2,
            num_workers=self.args.dataloader_num_workers,
        )
        fin_out = {}
        k = 0
        for batch in tqdm(dtel, desc="processing"):
            temp = correct_inputs(batch["images"], batch["nouns"])
            out = self.GD.predict(list(temp.values()), list(temp.keys()), 0.3, 0.25,)
            fin_out[k] = out
            # flushing intermediate output
            with open(os.path.join(self.args.output_metadata_folder, "temp_boxes.json"), 'w') as json_file:
                json.dump(fin_out, json_file, indent=4)
            json_file.close()
            k+=1
        del dtel, self.GD
        torch.cuda.empty_cache()
        
        f = open(os.path.join(self.args.output_metadata_folder, "metadata.jsonl"), "w")
        bbox_lst = [j for i in fin_out.values() for j in i]
        filname_lst = [j for j in filnames]
        noun_lst = [j for i in cn["nouns"].values() for j in i]
        caps_lst = [j for i in cn["captions"].values() for j in i]
        pretty_output(bbox_lst, filname_lst, noun_lst, caps_lst, f)
        f.close()

if __name__ == "__main__":
    # for pipeline 5
    args = parse_args()
    json_pth = config1["data_path"]
    generate_part_data(json_pth, args).forward()


