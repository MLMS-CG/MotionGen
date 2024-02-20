from torch.utils import data
import numpy as np
import mmap
from scipy.spatial.transform import Rotation as R

class Spactral(data.Dataset):
    def __init__(self, mode, datapath, nb_freqs, offset, size_window, std_mean=None, return_gender=False, rot_aug = False):
        # load dataset
        ## load length and gender data
        with open(datapath + "lengths.bin", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            self.lengths = np.frombuffer(mm, dtype=int)
        with open(datapath + "genders.bin", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            self.genders = np.frombuffer(mm, dtype=int)
        self.means_stds = std_mean
        self.sum_lengths = np.array(
            [np.sum(self.lengths[:i]) for i in range(len(self.lengths))]
        )
        ## load coefs data
        filename_dataset = datapath + "dataset.bin"
        self.dataset = np.memmap(
            filename_dataset, dtype="float32", mode="r"
        )
        filename_dataset = datapath + "rots.bin"
        self.rots = np.memmap(
            filename_dataset, dtype="float64", mode="r"
        )
        filename_dataset = datapath + "trans.bin"
        self.trans = np.memmap(
            filename_dataset, dtype="float64", mode="r"
        )
        filename_dataset = datapath + "tpose.bin"
        self.tpose = np.memmap(
            filename_dataset, dtype="float64", mode="r"
        ).reshape(-1,1024,3)

        self.rot_aug_mat = np.stack([R.from_rotvec([0,0,i*np.pi]).as_matrix() for i in np.arange(0,2,0.001)])
        self.rot_aug = rot_aug

        ## build indices
        self.nb_freqs = nb_freqs
        self.offset = offset
        self.size_window = size_window
        self.crop_len = self.size_window * self.nb_freqs * 3
        self.return_gender = return_gender

        self.gender_input = []
        self.chunkIndexStartFrame = []
        self.tpose_input = []
        for i in range(len(self.lengths)):
            current_length = self.lengths[i]
            for j in range(0, current_length, self.offset):
                if(j + self.size_window) < current_length:
                    offset = (self.sum_lengths[i] + j) * self.nb_freqs * 3
                    offset_rot_trans = (self.sum_lengths[i] + j) *  3
                    self.chunkIndexStartFrame.append([i, offset, offset_rot_trans])
                    self.gender_input.append(self.genders[i])
                    self.tpose_input.append(self.tpose[i])

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
            self.genders = self.gender_input[:train_length]
            self.tpose = self.tpose_input[:train_length]
            self.calStdMean(
                self.dataset[:self.sum_lengths[index]*self.nb_freqs*3]
            )
        else:
            self.lengths = val_length
            self.genders = self.gender_input[train_length:train_length+val_length]
            self.tpose = self.tpose_input[train_length:train_length+val_length]
            self.indexes = self.chunkIndexStartFrame[train_length:train_length+val_length]

    def calStdMean(self,data):
        data = np.array(data.reshape(-1, self.nb_freqs, 3))
        self.means_stds = [data.mean(0), data.std(0)]

    def __getitem__(self, idx):
        # get coefs for training
        selected = np.array(self.dataset[
            self.chunkIndexStartFrame[idx][1]:\
            self.chunkIndexStartFrame[idx][1] + self.crop_len 
        ]).reshape(self.size_window, self.nb_freqs, 3)
        rot = np.array(self.rots[
            self.chunkIndexStartFrame[idx][2]:\
            self.chunkIndexStartFrame[idx][2] + self.size_window*3
        ]).reshape(self.size_window, 1, 3)
        trans = np.array(self.trans[
            self.chunkIndexStartFrame[idx][2]:\
            self.chunkIndexStartFrame[idx][2] + self.size_window*3
        ]).reshape(self.size_window, 1, 3)
        # only trans in x-y space are initialized to origin point
        # this will make the initial foot z coordinate to 0
        trans[:,:,:2] = trans[:,:,:2] - trans[0,:,:2]

        return np.concatenate([(selected-self.means_stds[0])/self.means_stds[1], rot, trans], axis=1).astype(np.float32) \
        ,((self.tpose[idx]-self.means_stds[0])/self.means_stds[1]).astype(np.float32)

    def __len__(self):
        return self.lengths