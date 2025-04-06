import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pdb

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
            mssg = [{"role":"system", "content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},{"role":"user","content":cont2_q},{"role":"user","content":cont2_s},
                    {"role":"user","content":cont3_q}, {"role":"user","content":cont4_q}, {"role":"user","content":cont5_q},{"role":"user","content":cont6_q}]
        temp.append(mssg)
    return temp

class run_qwen(object):
    def __init__(self, args):
        self.device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained(args.llm_model, device_map="auto", cache_dir = args.cache_dir, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model, cache_dir = args.cache_dir)

    def get_summary(self, prt):
        # get summary
        txt2 = self.tokenizer.apply_chat_template(make_message_qwen(prt,0), tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(txt2, return_tensors="pt", padding=True, truncation=True, padding_side='left').to(self.device)
        model_inputs = {k: v.to(self.device) for k,v in model_inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, max_new_tokens=512)
        response1 = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [i.split('\n')[-1] for i in response1]
    
    def get_nouns(self, response1):
        # get nouns
        txt2 = self.tokenizer.apply_chat_template(make_message_qwen(response1, 1), tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(txt2, return_tensors="pt", padding=True, truncation=True, padding_side='left').to(self.device)
        model_inputs = {k: v.to(self.device) for k,v in model_inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, max_new_tokens=512)
        response2 = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [i.split('\n')[-1] for i in response2]

# if __name__ == "__main__":
#     prt = ["A cat and a dog playing in a park.", "Two busses travelling on street."]
#     args = {"model_name": "Qwen/Qwen2.5-7B-Instruct", "cache_dir": "/nfshomes/asarkar6/trinity/model_weights/"}
#     llm_obj = run_qwen(args)
#     r1 = llm_obj.get_summary(prt)
#     r2 = llm_obj.get_nouns(r1)
#     pdb.set_trace()
