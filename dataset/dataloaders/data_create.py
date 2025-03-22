import torch
import torchvision.transforms as v2
import numpy as np
from random import shuffle

class Trinitydata(object):
    def __init__(self):
        # read data
        self.x = np.load('/nfshomes/asarkar6/trinity/trinity_x1.npy', allow_pickle=True).reshape(-1, 1)
        self.y = np.load('/nfshomes/asarkar6/trinity/trinity_y1.npy', allow_pickle=True)
        self.ind_sl = list(range(1,self.x.shape[0]))
        shuffle(self.ind_sl)
        self.ind_l = list(range(1,self.x.shape[0]))
    
    def length(self):
        return self.x.shape[0]
    
    def forward(self, batch_size=32, shuffle=True, idx=0):
        num_batch = self.length()//batch_size
        batch = {}
        if shuffle:
            # shuffle data
            for i in range(0, num_batch):
                batch[i] = self.ind_sl[batch_size*i:batch_size*(i+1)]
        else:
            # shuffle data
            for i in range(0, num_batch):
                batch[i] = self.ind_sl[batch_size*i:batch_size*(i+1)]
            
        # create batches
        return self.x[batch[idx]], self.y[batch[idx]]

if __name__ == "__main__":
    data, label = Trinitydata().forward(4, True, 5)
    print(data.shape, label.shape)

