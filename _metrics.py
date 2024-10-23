

import json
import torch
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.enabled = False
exp_name = "binfile_onlytext_conditioned/"
path = "./save/" + exp_name

with open("preProcessing/default_options_dataset.json", "r") as outfile:
    opt = json.load(outfile)

if __name__=='__main__':
    dataset_path = "save/generated_result/"
    penetrates = []
    floats = []
    skates = []
    num_frames = []
    J_regressor = np.load("J_regressor.npy")

    for l in tqdm(range(21)):
        leftfoot = []
        rightfoot = []
        # act = np.load(dataset_path + 'act_'+str(i)+'.npy')
        meshes = np.load(dataset_path + 'meshes_'+str(l)+'.npy')
        meshes = meshes.reshape(32,90,6890,3)
        # tpose = np.load(dataset_path + 'tpose_'+str(i)+'.npy')
        for j in range(32):
            mesh = meshes[j]
            for k in range(90):
                joints = np.matmul(J_regressor, mesh[k])
                leftfoot.append(joints[10])
                rightfoot.append(joints[11])

            for i in range(len(mesh)-1):
                if np.min(mesh[i,[6715, 6736, 6762]])<0.005 and np.min(mesh[i+1,[6715, 6736, 6762]])<0.005:
                    skates.append(np.sqrt(np.sum((leftfoot[i]-leftfoot[i+1])**2)))
                if np.min(mesh[i,[3316, 3318, 3336, 3344, 3362]])<0.005 and np.min(mesh[i+1,[3316, 3318, 3336, 3344, 3362]])<0.005:
                    skates.append(np.sqrt(np.sum((rightfoot[i]-rightfoot[i+1])**2)))
            lowest = mesh.min(1)
            num_frames.append(len(mesh))
            penetrates.append(lowest[lowest<-0.005].sum())
            floats.append(lowest[lowest>0.005].sum())
    print(1)