from torch.utils import data
import numpy as np
import mmap
from scipy.spatial.transform import Rotation as R

class Spactral(data.Dataset):
    def __init__(self, mode, datapath, nb_freqs, offset, size_window, std_mean=None, return_gender=False, rot_aug = False):
        # load dataset
        ## load length and gender data
        self.mode = mode
        with open(datapath + "lengths.bin", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            self.lengths = np.frombuffer(mm, dtype=int)
        self.means_stds = std_mean
        self.sum_lengths = np.array(
            [np.sum(self.lengths[:i]) for i in range(len(self.lengths))]
        )
        ## load coefs data
        self.actions = np.load(datapath + "action.npy")
        filename_dataset = datapath + "dataset.bin"
        self.dataset = np.memmap(
            filename_dataset, dtype="float32", mode="c"
        )
        filename_dataset = datapath + "rots.bin"
        self.rots = np.memmap(
            filename_dataset, dtype="float64", mode="c"
        )
        filename_dataset = datapath + "trans.bin"
        self.trans = np.memmap(
            filename_dataset, dtype="float64", mode="c"
        )
        filename_dataset = datapath + "tpose.bin"
        self.tpose = np.memmap(
            filename_dataset, dtype="float64", mode="c"
        ).reshape(-1,1024,3)

        ## build indices
        self.nb_freqs = nb_freqs
        self.offset = offset
        self.size_window = size_window
        self.crop_len = self.size_window * self.nb_freqs * 3
        self.action_input = []
        self.chunkIndexStartFrame = []
        for i in range(len(self.lengths)):
            current_length = self.lengths[i]
            for j in range(0, current_length, self.offset):
                if(j + self.size_window) < current_length:
                    offset = (self.sum_lengths[i] + j) * self.nb_freqs * 3
                    offset_rot_trans = (self.sum_lengths[i] + j) *  3
                    self.chunkIndexStartFrame.append([i, offset, offset_rot_trans])
                    if self.actions[i] in a2l.keys():
                        self.action_input.append(a2l[self.actions[i]])
                    else:
                        self.action_input.append(-1)

        # slice the dataset
        total_length = len(self.chunkIndexStartFrame)
        # shuffled_index = np.arange(total_length)
        # np.random.shuffle(shuffled_index)
        shuffled_index = np.load(datapath + "shuffleindex.npy")
        self.chunkIndexStartFrame = [self.chunkIndexStartFrame[i] for i in shuffled_index]
        self.action_input = np.array(self.action_input)[shuffled_index]

        self.train_length = total_length
        val_length = int(0.2 * total_length)
        
        self.mean = -1
        self.std = -1
        # split dataset(split indices)
        if mode == "train":
            self.lengths = self.train_length
            # self.indexes = self.chunkIndexStartFrame[:train_length]
            self.actions = self.action_input[:self.train_length]
            self.len = 160000
            # index = self.chunkIndexStartFrame[self.train_length][0]-1
            self.calStdMean(
                # self.dataset[:self.sum_lengths[index]*self.nb_freqs*3]
                self.dataset
            )
        else:
            self.lengths = val_length
            self.actions = self.action_input[self.train_length:self.train_length+val_length]
            self.len = 20000
            # self.indexes = self.chunkIndexStartFrame[train_length:train_length+val_length]
        self.new_epoch()

    def mirrow_rotation(self, matrix):
        theta_z = np.arctan2(matrix[1,0], matrix[0,0])
        theta_y = np.arctan2(-matrix[2,0], np.sqrt(matrix[2,1]**2+matrix[2,2]**2))
        cosz, sinz = np.cos(-theta_z), np.sin(-theta_z)
        cosy, siny = np.cos(-2*theta_y), np.sin(-2*theta_y)

        m_matrixz = [[cosz, -sinz, 0], [sinz, cosz, 0], [0,0,1]]
        m_matrixy = [[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]]
        matrix = np.matmul(m_matrixz, matrix)
        matrix = np.matmul(m_matrixy, matrix)
        matrix = np.matmul(m_matrixz, matrix)
        return matrix

    def mirrow_all(self, coefs, trans, rots):
        trans_array = np.array([[-1,0,0],[0,1,0],[0,0,1]]).astype(np.float32)
        trans = np.matmul(trans, trans_array)
        coefs = np.matmul(coefs, trans_array)
        rots = np.stack([
                    R.from_matrix
                    (
                        self.mirrow_rotation(
                            R.from_rotvec(mat).as_matrix(), 
                                )
                    ).as_rotvec()
                    for mat in rots
                ])
        return coefs, trans, rots

    def calStdMean(self,data):
        data = np.array(data.reshape(-1, self.nb_freqs, 3))
        # mean
        # mean = data.mean(0)
        # self.means_stds = [np.zeros_like(mean), np.zeros_like(mean)] # mean std
        # self.means_stds[0][:,1:] = mean[:,1:] # because of mirror based on yz plane, mean of x is zero
        # # std
        # self.means_stds[1][:,1:] = data.std(0)[:,1:]
        # self.means_stds[1][:,0] = np.concatenate([data[:,:,0],-data[:,:,0]],axis=0).std(0)
        self.means_stds = [data.mean(0), data.std(0)]

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        # get coefs for training
        selected = np.array(self.dataset[
            self.chunkIndexStartFrame[idx][1]:\
            self.chunkIndexStartFrame[idx][1] + self.crop_len 
        ]).reshape(self.size_window, self.nb_freqs, 3)
        rot = np.array(self.rots[
            self.chunkIndexStartFrame[idx][2]:\
            self.chunkIndexStartFrame[idx][2] + self.size_window*3
        ]).reshape(self.size_window, 3)
        trans = np.array(self.trans[
            self.chunkIndexStartFrame[idx][2]:\
            self.chunkIndexStartFrame[idx][2] + self.size_window*3 
        ]).reshape(self.size_window, 3)
        # only trans in x-y space are initialized to origin point
        # this will make the initial foot z coordinate to 0
        trans[:,:2] = trans[:,:2] - trans[0,:2]
        
        # flag_mirrow = np.random.choice([0,1])
        # if flag_mirrow:
        #     selected, trans, rot = self.mirrow_all(selected, trans, rot)
        rot = rot.reshape(self.size_window, 1, 3)
        trans = trans.reshape(self.size_window, 1, 3)

        return np.concatenate([(selected-self.means_stds[0])/self.means_stds[1], rot, trans], axis=1).astype(np.float32) \
        ,((self.tpose[self.chunkIndexStartFrame[idx][0]]-self.means_stds[0])/self.means_stds[1]).astype(np.float32) \
        , self.action_input[idx]

    def __len__(self):
        return len(self.indexes)
    
    def new_epoch(self):
        self.indexes  = []
        used = [0,1,2,3,4,5,6,7]
        for i in used:
            indexes = np.where(self.actions==i)[0]
            if self.mode=="eval":
                indexes += self.train_length
            self.indexes.append(np.random.choice(indexes, int(self.len/len(used))))
        self.indexes=np.concatenate(self.indexes, axis=0)

a2l = {
    "walk": 0,
    "jump": 1,
    "run": 2,
    "sit": 3,
    "stretch": 4,
    "throw": 5,
    "kick": 6,
    "gesture": 7,
}