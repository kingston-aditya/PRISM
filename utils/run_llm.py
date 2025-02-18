import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def make_summary_phi3(prt):
    cont1_q = "Question - Summarize the text to a maximum of 12 words and output the nouns in the Answer: Artificial intelligence is rapidly changing many industries, providing new opportunities, challenges, and innovations. It's transforming the way we work, live, and interact."
    cont1_a = "Answer - Artificial intelligence is changing industries, providing opportunities, transforming work and interaction. \n Nouns - intelligence, industries, opportunities, work, interaction."
    cont2_q = "Question - Summarize the text to a maximum of 12 words and output the nouns in the Answer: Quantum computing represents a new frontier in computing technology, offering solutions to complex problems that traditional computers struggle with, such as cryptography and optimization."
    cont2_a = "Answer - Quantum computing offers solutions to problems traditional computers struggle with, like cryptography. \n Nouns - computing, solutions, problems, computers, cryptography."
    cont3_q = "Question - Summarize the text to a maximum of 12 words and output the nouns in the Answer: The novel tells the story of a young woman who embarks on an adventure to discover her true identity, facing numerous challenges and learning profound life lessons along the way."
    cont3_a = "Answer - A young woman embarks on an adventure to discover her true identity. \n Nouns - woman, adventure, identity."
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}, 
                {"role": "user", "content": cont1_q},
                {"role": "assistant", "content": cont1_a},
                {"role": "user", "content": cont2_q},
                {"role": "assistant", "content": cont2_a},
                {"role": "user", "content": cont3_q},
                {"role": "assistant", "content": cont3_a}
                ]
    messages.append({"role": "user", "content": "Question - Summarize the text to a maximum of 12 words and output the nouns in the Answer:" + prt})
    return messages

def make_summary_qwen(prt,typ):
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
    return mssg

class run_phi3(object):
    def __init__(self):
        self.device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map=self.device, torch_dtype="auto", trust_remote_code=True,)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    def forward(self, prt):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        gen_args = {"max_new_tokens": 500, "return_full_text": False, "temperature": 0.0, "do_sample": False, }
        output = pipe(make_summary_phi3(prt), **gen_args)
        return output[0]['generated_text']

class run_qwen(object):
    def __init__(self):
        self.device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map=self.device, cache_dir = "/nfshomes/asarkar6/trinity/model_weights/", torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir = "/nfshomes/asarkar6/trinity/model_weights/")
    
    def forward(self, prt):
        # get summary
        txt1 = self.tokenizer.apply_chat_template(make_summary_qwen(prt,0), tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([txt1], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response1 = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # get nouns
        txt2 = self.tokenizer.apply_chat_template(make_summary_qwen(response1, 1), tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([txt2], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response2 = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response1, response2

if __name__ == "__main__":
    prt = "A serene lake at sunset, surrounded by towering, snow-capped mountains. The water reflects the orange and pink sky, while a small wooden boat drifts gently. Pine trees line the shore, their dark green needles contrasting with the warm hues of the evening. The air feels calm and crisp."
    llm_obj = run_qwen()
    r1, r2 = llm_obj.forward(prt)
    print("1", r1)
    print("2", r2)
