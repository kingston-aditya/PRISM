import torch
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore")

from cc3m_dataloader import return_cc3_train_dataset
from sharegpt_dataloader import Caps_Nouns_Filenames
from utils import correct_inputs, pretty_output, dynamic_collate, dynamic_collate_1

from config import get_config
config1 = get_config()

import sys
sys.path.insert(1, config1["repo_path"])
from utils import run_gd, run_llm, run_mllm

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
        "--dataloader_num_workers",
        type=int,
        default=12,
        help="Number of workers",
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
        # Task 1 - Save the images.
        cn = {"captions": [], "nouns": []}
        dtel = DataLoader(
            self.pil_dataset,
            shuffle=False,
            batch_size = self.args.batch_size,
            collate_fn=dynamic_collate,
            num_workers=self.args.dataloader_num_workers,
        )
        print(len(dtel))
        k = 0
        img_dataset = {"file_name":[], "images": []}
        fil_name = []
        for batch in dtel:
            temp = []
            for img in batch:
                img.save(os.path.join(self.args.output_img_folder, str(k)+".png"))
                temp.append(os.path.join(self.args.output_img_folder, str(k)+".png"))
                img_dataset["file_name"].append(os.path.join(self.args.output_img_folder, str(k)+".png"))
                img_dataset["images"].append(img)
                k += 1
            fil_name.append(temp)

        # Task 2 - Get detailed captions
        self.qwen_model = run_mllm.run_quen2_vl(self.args)
        caps_lst = []
        for batch in fil_name:
            caps = self.qwen_model.forward(batch)
            caps_lst.append(caps)
        del self.qwen_model

        # Task 3 - get the nouns and captions.
        cn = {"captions": [], "nouns": []}
        self.llm_obj = run_llm.run_qwen(self.args)
        for batch in tqdm(caps_lst, desc="Processing"):
            short_caps = self.llm_obj.get_summary(batch)
            nouns = self.llm_obj.get_nouns(short_caps)
            cn["captions"].append(short_caps)
            cn["nouns"].append(nouns)
        del self.llm_obj, dtel
        torch.cuda.empty_cache()

        # Task 4 - get the objects
        self.GD = run_gd.GDINO(args)
        get_caps_nouns_filenames = Caps_Nouns_Filenames([j for i in cn["nouns"] for j in i], [k for k in img_dataset["images"]])
        dtel = DataLoader(
            get_caps_nouns_filenames,
            shuffle=False,
            batch_size = self.args.batch_size,
            collate_fn=dynamic_collate_1,
            num_workers=self.args.dataloader_num_workers,
        )
        fin_out = []
        for batch in tqdm(dtel, desc="processing"):
            temp = correct_inputs(batch["images"], batch["nouns"])
            out = self.GD.predict(list(temp.values()), list(temp.keys()), 0.3, 0.25,)
            fin_out.append(out)
        
        f = open(os.path.join(self.args.output_metadata_folder, "metadata.jsonl"), "w")
        bbox_lst = [j for i in fin_out for j in i]
        filname_lst = [j for j in img_dataset["file_name"]]
        noun_lst = [j for i in cn["nouns"] for j in i]
        caps_lst = [j for i in cn["captions"] for j in i]
        pretty_output(bbox_lst, filname_lst, noun_lst, caps_lst, f)
        f.close()

if __name__ == "__main__":
    # for pipeline 5
    args = parse_args()
    json_pth = "path_to_cc3m"
    generate_real_data(json_pth, args).forward()


