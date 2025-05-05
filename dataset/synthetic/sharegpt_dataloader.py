from torch.utils.data import Dataset
import json

def ShareGPT(json_pth, batch_size, num1, num2):
    # read json objects
    f = open(json_pth, "r")
    json_obj = json.load(f)
    f.close()

    # get the captions
    caps = []
    num2 = min(len(json_obj), num2)
    for i in range(num1, num2):
        prts = json_obj[i]["conversations"][1]['value'].split("\n\n")
        for j in prts:
            caps.append(j)
        if len(caps) >= num2-num1:
            break
    
    # create batches
    t = {}
    for i in range(len(caps)//batch_size):
        t[i] = caps[batch_size*i:batch_size*(i+1)]
    
    if len(caps)%batch_size != 0:
        t[i+1] = caps[batch_size*(i+1):num2]

    return list(t.values())

