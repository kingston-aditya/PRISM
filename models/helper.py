import json
import numpy as np
import itertools

def helper(path_name):
    f = open(path_name)
    js = json.load(f)

    c = 0

    for idx in range(len(js)):
        try:
            js[idx]["file_name"] = js[idx].pop("img_pth")
            js[idx]["prompt"] = js[idx].pop("caps")
            
            if "bbox" in js[idx].keys():
                js[idx]["object"] = js[idx].pop("bbox")
                c = c+1
        except:
            print(f"error at index {idx}")

    with open(path_name, 'w') as json_file:
        json.dump(js, json_file, indent=4)
    
    print(c/len(js))

def join_json(path_names):
    a = []
    for path_name in path_names:
        f = open(path_name)
        js = json.load(f)

        a.append(js)
    a_new = list(itertools.chain.from_iterable(a))

    with open("/nfshomes/asarkar6/trinity/trinity-data-real-combined.json", 'w') as json_file:
        json.dump(a_new, json_file, indent=4)
    


if __name__ == "__main__":
    path_list = ["/nfshomes/asarkar6/trinity/trinity-data-real.json", "/nfshomes/asarkar6/trinity/trinity-data-real2.json", "/nfshomes/asarkar6/trinity/trinity-data-real3.json"]
    join_json(path_list)
    

        


