def get_config():
    return {
        "repo_path": "/data/home/saividyaranya/PRISM/",
        "input_data_dir": "/fsx/mrs_shlok_sai/cc12m_v2/",
        "llm_model": "Qwen/Qwen2.5-72B-Instruct",
        "mllm_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "cache_dir": "/data/home/saividyaranya/PRISM/cached_folder_real",
        "batch_size": 512,
        "dataloader_num_workers": 1,
        "is_sdxl": "False",
        "start_len": 3_000_000,
        "end_len": 4_500_000,
        "output_metadata_folder": "/data/home/saividyaranya/PRISM/cached_folder_real/metadata_folder",
        "output_img_folder": "/data/home/saividyaranya/PRISM/cached_folder_real/images/",
        "job_id": 1 
    }
args = get_config()

# this script generates captions and nouns for synthetic
# data. Part of synthetic pipeline.
import os
import torch 
from transformers import AutoTokenizer
import pdb
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json
import time

def make_message_qwen(batch,typ):
    temp = []
    for prt in batch:
        if typ == 0:
            cont1_q = "Question - Improve the language of the text. It should contain all the objects from the input prompt." + "\n\n Prompt -" + prt + "\n\n Output -"
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

    # load the captions
    f = open(os.path.join(args["output_metadata_folder"], "temp_caps" + str(args["job_id"]) + ".json"))
    cn = json.load(f)

    k=0
    llm_obj = run_qwen(args)
    for item in tqdm(list(cn["captions"].values()), desc="Processing"):
        r1 = llm_obj.forward(item, 0)
        r2 = llm_obj.forward(r1, 1)
        cn["captions"][k] = r1
        cn["nouns"][k] = r2
        k+=1
    del llm_obj
    
    # save dataset
    with open(os.path.join(args["output_metadata_folder"], "temp_caps"+ str(args["job_id"]) +".json"), 'w') as json_file:
        json.dump(cn, json_file, indent=4)
    json_file.close()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total RUNTIME is {end_time - start_time}")
    
    
    
