import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def make_summary(prt):
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

class run_phi3(object):
    def __init__(self):
        self.device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map=self.device, torch_dtype="auto", trust_remote_code=True,)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    def forward(self, prt):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        gen_args = {"max_new_tokens": 500, "return_full_text": False, "temperature": 0.0, "do_sample": False, }
        output = pipe(make_summary(prt), **gen_args)
        return output[0]['generated_text']

if __name__ == "__main__":
    prt = "The global economy is facing significant challenges as inflation rates rise, unemployment increases, and markets remain volatile. Governments and central banks are taking action to stabilize the economy, but recovery is expected to take time."
    llm_obj = run_phi3()
    print(llm_obj.forward(prt))
