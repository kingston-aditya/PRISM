import os
import torch 
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

import json
import time
import glob

import pdb

# get all the args
from config import get_config
args = get_config()

def make_message_qwen(batch, num):
    temp = []
    for prt in batch:
        cont2_q = "An instruction related to image editing and a number, say N, are provided. Your task is to paraphrase it N times without modifying the nouns in the prompt."
        cont2_s = "Here are three examples."
        cont3_q = "Give 2 paraphrased prompts for the following prompt: The book is on the table. \n\n Output - The table has a book on it. \n There's a book resting on the table."
        cont4_q = "Give 3 paraphrased prompts for the following prompt: She opened the umbrella in the rain. \n\n Answer - She used an umbrella when it started raining. \n In the rain, she popped open her umbrella. \n An umbrella was opened by she in the rain."
        cont5_q = "Give 2 paraphrased prompts for the following prompt: He put the keys in his pocket. \n\n Answer - He slipped the keys into his pocket. \n The keys went into his pocket."
        cont6_q = "Do NOT give any explanation. Do NOT output the same input prompt.  Prompt - " + prt + ", Number - " + str(num) + "\n\n Answer - "
        mssg = [{"role":"system", "content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},{"role":"user","content": cont2_q + "\n\n" + cont2_s + "\n\n" + cont3_q + "\n\n" + cont4_q + "\n\n" + cont5_q + "\n\n" + cont6_q}]
        temp.append(mssg)
    return temp

def make_batch(main_prt, num):
    a = [main_prt[i:i+num] for i in range(0,len(main_prt),num)]
    return a

class run_qwen(object):
    def __init__(self, args):
        self.model = LLM(model=args["llm_model"], tensor_parallel_size=torch.cuda.device_count(), download_dir=args["cache_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(args["llm_model"], cache_dir = args["cache_dir"])
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    def forward(self, prt, num):
        # get summary or nouns
        txt2 = self.tokenizer.apply_chat_template(make_message_qwen(prt, num), tokenize=False, add_generation_prompt=True)
        outputs = self.model.generate(txt2, self.sampling_params)
        temp = []
        for output in outputs:
            temp.append(output.outputs[0].text)
        return temp 

if __name__ == "__main__":
    start_time = time.time()

    # run the qwen model
    llm_obj = run_qwen(args) 
    for name in glob.glob(os.path.join(args["data_path"],"metadata*.jsonl")):
        main_prt = []; full_prt = []
        with open(os.path.join(args["data_path"], name), "r") as f:
            for line in f:
                temp = json.loads(line.strip())
                prt = temp["prompt"]
                main_prt.append(prt)
                full_prt.append(temp)

        # make batches
        prompt_loader = make_batch(main_prt, args["batch_size"])
        r1 = []
        for item in tqdm(prompt_loader, desc="Augmenting"):
            try:
                r1.extend(llm_obj.forward(item, args["copy_num"]))
            except:
                print("!! Something went wrong !!")
                r1.extend(item)

        # write it on json file
        with open(os.path.join(args["data_path"], name.split(".")[0]+"_n"+".jsonl"), "w") as f:
            for idx, item in enumerate(r1):
                mini_prts = item.split("\n")
                f.write(json.dumps(full_prt[idx])+"\n")
                for j in mini_prts:
                    full_prt[idx]["prompt"] = j
                    f.write(json.dumps(full_prt[idx])+"\n")

        torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total RUNTIME is {end_time - start_time}")
    
    
    