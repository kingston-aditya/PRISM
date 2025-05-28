import math
import numpy as np
import json

def correct_inputs(imgs, txts):
    expanded_imgs_list = []
    expanded_txts_list = []
    for i in range(len(txts)):
        for j in txts[i].split(","):
            # temp[j] = imgs[i]
            expanded_imgs_list.append(imgs[i])
            expanded_txts_list.append(j)
    return expanded_imgs_list, expanded_txts_list

def GD_batcher(imgs, txts, batch_size):
    # create batches
    t = []; t1 = []
    s = 0
    for i in range(len(txts)//batch_size):
        # t[i] = txts[batch_size*i:batch_size*(i+1)]
        t.append(txts[batch_size*i:batch_size*(i+1)])
        # t1[i] = imgs[batch_size*i:batch_size*(i+1)]
        t1.append(imgs[batch_size*i:batch_size*(i+1)])
        s+=1

    if len(txts)%batch_size != 0:
        # t[s] = txts[batch_size*(s):len(txts)]
        t.append(txts[batch_size*(s):len(txts)])
        # t1[s] = imgs[batch_size*(s):len(imgs)]
        t1.append(imgs[batch_size*(s):len(imgs)])

    # return list(t.values()), list(t1.values())
    return t, t1

def pretty_output(bbox_lst, filname_lst, noun_lst, cap_lst, f):
    k = 0
    for i in range(len(noun_lst)):
        # end_idx = min(len(noun_lst[i].split(",")), 3)
        end_idx =len(noun_lst[i].split(","))
        object_temp = bbox_lst[k:k+end_idx]
        atema = []
        for item in object_temp:
            if len(item['scores']) !=0:
                xmin = math.ceil(np.asarray(item["boxes"])[0][0])
                ymin = math.ceil(np.asarray(item["boxes"])[0][1])
                xmax = math.ceil(np.asarray(item["boxes"])[0][2])
                ymax = math.ceil(np.asarray(item["boxes"])[0][3])
                labels = item["labels"][0]
                filname = str(filname_lst[i])
                atema.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels, "img_pth": filname})
            else:
                pass

        # import pdb; pdb.set_trace()
        temp = {"file_name": filname_lst[i], "prompt": cap_lst[i], "object": atema[:3]}
        k += end_idx
        f.write(json.dumps(temp) + '\n')