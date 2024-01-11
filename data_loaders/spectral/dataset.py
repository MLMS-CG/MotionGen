from torch.utils import data
import numpy as np
import mmap
import types

class Spactral(data.Dataset):
    def __init__(self, mode, datapath, nb_freqs, offset, size_window):
        # load dataset
        ## load length data
        with open(datapath + "lengths.bin", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            self.lengths = np.frombuffer(mm, dtype=int)
        self.means_stds = None
        self.sum_lengths = np.array(
            [np.sum(self.lengths[:i]) for i in range(len(self.lengths))]
        )
        ## load coefs data
        filename_dataset = datapath + "dataset.bin"
        self.dataset = np.memmap(
            filename_dataset, dtype="float32", mode="r"
        )
        ## build indices
        self.nb_freqs = nb_freqs
        self.offset = offset
        self.size_window = size_window
        self.crop_len = self.size_window * self.nb_freqs * 3

        self.chunkIndexStartFrame = []
        for i in range(len(self.lengths)):
            current_length = self.lengths[i]
            for j in range(0, current_length, self.offset):
                if(j + self.size_window) < current_length:
                    offset = (self.sum_lengths[i] + j) * self.nb_freqs * 3
                    self.chunkIndexStartFrame.append([i, offset])

        # slice the dataset
        total_length = len(self.chunkIndexStartFrame)
        train_length = int(0.8 * total_length)
        val_length = int(0.2 * total_length)
        
        self.mean = -1
        self.std = -1
        # split dataset(split indices)
        if mode == "train":
            self.lengths = train_length
            self.indexes = self.chunkIndexStartFrame[:train_length]
            index = self.chunkIndexStartFrame[train_length][0]-1
            self.calStdMean(
                self.dataset[:self.sum_lengths[index]*self.nb_freqs*3]
            )
        else:
            self.lengths = val_length
            self.indexes = self.chunkIndexStartFrame[train_length:train_length+val_length]

    def calStdMean(self,data):
        data = np.array(data.reshape(-1, self.nb_freqs, 3))
        self.means_stds = [data.mean((0,1)), data.std((0,1))]

    def __getitem__(self, idx):
        # get coefs for training
        selected = np.array(self.dataset[
            self.chunkIndexStartFrame[idx][1]:\
            self.chunkIndexStartFrame[idx][1] + self.crop_len 
        ]).reshape(self.size_window, self.nb_freqs, 3)
        return selected

    def __len__(self):
        # return len
        return self.lengths