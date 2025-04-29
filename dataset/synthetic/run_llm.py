# this script generates captions and nouns for synthetic
# data. Part of synthetic pipeline.

import os
import time

# get all the args
import argparse
from config import get_config
args = get_config()

def parse_args():
    parser = argparse.ArgumentParser(description="Use argparse for three params.")
    parser.add_argument('--start_len', type=int, help='Start len')
    parser.add_argument('--end_len', type=int, help='End len')
    parser.add_argument('--job_id', type=int, help='job id')

    fixn_args = parser.parse_args()
    return fixn_args

from sharegpt_dataloader import ShareGPT

import torch 
from transformers import AutoTokenizer
import pdb
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json

def make_message_qwen(batch,typ):
    temp = []
    for prt in batch:
        if typ == 0:
            cont1_q = "Question - Summarize the text to a maximum of 30 words. It should contain as many objects as possible from the input prompt." + "\n\n Prompt -" + prt + "\n\n Summary -"
            mssg = [{"role":"system", "content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},{"role":"user","content":cont1_q}]
        else:
            cont2_q = "An image is generated using the input prompt. Output objects that can be present in the generated image and there is no overlap between the bounding boxes on those objects. Objects and their adjectives should be from the input prompt and must be in less than 4 words. Abstract nouns should NOT be given as output."
            cont2_s = "Here are two examples."
            cont3_q = "Prompt - An brown dog is playing with a black cat in a lush green mountainous background. \n\n Answer - brown dog, black cat \n\n Explanation - It is easy to make bounding boxes on dog and cat, but it's not possible to make it on a background."
            cont4_q = "Prompt - Happy horses and sad zebras are standing in a green grass and expressing sadness with activity. \n\n Answer - horses, zebras \n\n Explanation - It is easy to make bounding boxes on horses and zebra, but it's not possible to make it on a background. Also Sadness, sad and happy and activity are abstract nouns, hence not considered."
            cont5_q = "Prompt - A green cat is going towards a blue light pole which is on a park. \n\n Answer - green cat, blue light pole \n\n Explanation - cat and light pole are objects. Park is not given as output because the bounding box on park might cover the whole image and that is not desirable."
            cont6_q = "Do NOT output explanatin. Prompt - " + prt + "\n\n Answer - "
            mssg = [{"role":"system", "content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},{"role":"user","content": cont2_q + "\n\n" + cont2_s + "\n\n" + cont3_q + "\n\n" + cont4_q + "\n\n" + cont5_q + "\n\n" + cont6_q}]
        temp.append(mssg)
    return temp

class run_qwen(object):
    def __init__(self, args):
        self.model = LLM(model=args["llm_model"], tensor_parallel_size=torch.cuda.device_count(), download_dir=args["cache_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(args["llm_model"], cache_dir = args["cache_dir"])
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    def forward(self, prt, typ):
        # get summary or nouns
        txt2 = self.tokenizer.apply_chat_template(make_message_qwen(prt, typ), tokenize=False, add_generation_prompt=True)
        outputs = self.model.generate(txt2, self.sampling_params)
        temp = []
        for output in outputs:
            temp.append(output.outputs[0].text)
        return temp 

if __name__ == "__main__":
    start_time = time.time()
    fixn_args = parse_args()

    cn = {"captions": {}, "nouns": {}}
    caps_dataset = ShareGPT(args["data_path"], args["batch_size"], fixn_args.start_len, fixn_args.end_len)

    k=0
    llm_obj = run_qwen(args)
    for i, item in enumerate(tqdm(caps_dataset, desc="Processing")):
        if i == len(caps_dataset) - 1:
            break
        print("something in for loop")
        r1 = llm_obj.forward(item, 0)
        r2 = llm_obj.forward(r1, 1)
        cn["captions"][k] = r1
        cn["nouns"][k] = r2
        k+=1
    print("end of for loop in run llm")
    del llm_obj
    # save dataset
    output_metadata_folder = args["output_metadata_folder"]
    temp_caps_path = os.path.join(output_metadata_folder, "temp_caps"+str(fixn_args.job_id)+".json")
    # Create the output metadata folder if it doesn't exist
    if not os.path.exists(output_metadata_folder):
        os.makedirs(output_metadata_folder)
    with open(temp_caps_path, 'w') as json_file:
    # with open(os.path.join(args["output_metadata_folder"], "temp_caps.json"), 'w') as json_file:
        json.dump(cn, json_file, indent=4)
    # json_file.close()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total RUNTIME from run llm is {end_time - start_time}")
    
    
    
