import torch
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
import argparse
import time
import json

import warnings
warnings.filterwarnings("ignore")

from cc3m_dataloader import return_cc3_train_dataset
from sharegpt_dataloader import Caps_Nouns_Filenames
from utils import correct_inputs, pretty_output, dynamic_collate_3, dynamic_collate_1

from config import get_config
config1 = get_config()

import sys
sys.path.insert(1, config1["repo_path"])
from utilities import run_gd, run_llm, run_mllm
import pdb 

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
        "--mllm_model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
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
        "--dataloader_num_workers",
        type=int,
        default=12,
        help="Number of workers",
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/finale_data/images",
        help="Batch size",
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

class generate_real_data(object):
    def __init__(self, args):
        self.pil_dataset = return_cc3_train_dataset(args)
        self.args = args

    def forward(self):
        start_time = time.time()
        # Task 1 - Read the images.
        cn = {"captions": {}, "nouns": {}}

        indices = list(range(self.args.start_len, self.args.end_len)) 
        subset_data = Subset(self.pil_dataset, indices)

        dtel = DataLoader(
            subset_data,
            shuffle=False,
            batch_size = self.args.batch_size,
            collate_fn=dynamic_collate_3,
            num_workers=self.args.dataloader_num_workers,
        )
        print("Number of batches", len(dtel), "Total size", len(dtel)*self.args.batch_size)
        
        k = 0
        img_dataset = {"file_name":{}, "images": {}}
        for batch in dtel:
            for img in batch:
                img.save(os.path.join(self.args.output_img_folder, str(k)+".png"))
                img_dataset["file_name"][k] = os.path.join(self.args.output_img_folder, str(k)+".png")
                img_dataset["images"][k] = img
                k += 1
        end_time = time.time()
        print(f"Total runtime of the TASK 1 is {end_time - start_time}") 
        pdb.set_trace()

        # Task 2 - Get detailed captions
        self.qwen_model = run_mllm.run_florence(self.args)
        cn = {"captions": {}, "nouns": {}}
        k=0
        for batch in dtel:
            caps = self.qwen_model.forward(batch)
            cn["captions"][k] = caps
            k+=1
        del self.qwen_model
        torch.cuda.empty_cache()
        end_time = time.time()
        print(f"Total runtime of the TASK 2 is {end_time - start_time}") 

        # Task 3 - get the nouns and captions.
        k=0
        self.llm_obj = run_llm.run_qwen(self.args)
        for batch in tqdm(list(cn["captions"].values()), desc="Processing"):
            nouns = self.llm_obj.get_nouns(batch)
            cn["nouns"][k] = nouns
            k+=1
        del self.llm_obj
        torch.cuda.empty_cache()
        end_time = time.time()
        print(f"Total runtime of the TASK 3 is {end_time - start_time}") 

        # flushing intermediate output
        with open(os.path.join(self.args.output_metadata_folder, "temp_captions.json"), 'w') as json_file:
            json.dump(cn, json_file, indent=4)

        # Task 4 - get the objects
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
        k=0
        for batch in tqdm(dtel, desc="processing"):
            temp = correct_inputs(batch["images"], batch["nouns"])
            out = self.GD.predict(list(temp.values()), list(temp.keys()), 0.3, 0.25,)
            fin_out[k] = out
            k+=1
        end_time = time.time()
        print(f"Total runtime of the TASK4 is {end_time - start_time}") 

        f = open(os.path.join(self.args.output_metadata_folder, "metadata.jsonl"), "w")
        bbox_lst = [j for i in fin_out.values() for j in i]
        filname_lst = [j for j in img_dataset["file_name"].values()]
        noun_lst = [j for i in cn["nouns"].values() for j in i]
        caps_lst = [j for i in cn["captions"].values() for j in i]
        pretty_output(bbox_lst, filname_lst, noun_lst, caps_lst, f)
        f.close()
        end_time = time.time()
        print(f"Total runtime is {end_time - start_time}") 

if __name__ == "__main__":
    # for pipeline 5
    args = parse_args()
    generate_real_data(args).forward()


