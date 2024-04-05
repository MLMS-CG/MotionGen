import numpy as np
import os
import matplotlib.pyplot as plt

dataPath = "/home/kxue/Work/MotionGen/AMASS/"

datasets = [
    "ACCAD/",
    "BMLhandball/",
    "BMLmovi/",
    "BioMotionLab_NTroje/",
    "CMU/",
    "DanceDB/",
    "DFaust_67",
    "EKUT/",
    "Eyes_Japan_Dataset/",
    "GRAB/",
    "HUMAN4D/",
    "MPI_HDM05/",
    "HumanEva/",
    "KIT/",
    "MPI_mosh/",
    "MPI_Limits/",
    "SFU/",
    "SOMA/",
    "SSM_synced/",
    "TCD_handMocap/",
    "TotalCapture/",
    "Transitions_mocap/",
    "WEIZMANN/",
]

shapes = {}

for dataset in datasets:
    folders = os.listdir(dataPath + dataset)
    for folder in folders:
        if not os.path.isdir(dataPath + dataset + folder):
            continue
        files = os.listdir(dataPath + dataset + folder)
        shapesPerFolder = set()
        for file in files:
            data = np.load(dataPath + dataset + '/' + folder + '/' + file)
            if "poses" not in data.files:
                continue
            shape = tuple(data.f.betas)
            if shape not in shapes.keys():
                shapes[shape] = {"gender":[],"motion":[], "frames":[]}
            shapes[shape]["gender"].append(data.f.gender)
            shapes[shape]["motion"].append(dataset + '/' + folder + '/' + file)
            shapes[shape]["frames"].append(data.f.poses.shape[0])

print(1)

# visualize input data
import mmap
from scipy.spatial.transform import Rotation as R
import trimesh

with open("data/evecs_4096.bin", "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    evecs = np.frombuffer(
        mm[:], dtype=np.float32).reshape(6890, 4096)
    
path_faces = "data/faces.bin"

with open(path_faces, "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    smpl_faces = np.frombuffer(mm[:], dtype=np.intc).reshape(-1, 3)
    
evecs = evecs[:, :1024]
mean, std = self.means_stds
mean = mean.cpu().numpy()
std = std.cpu().numpy()
def visualize(data):
    result = data.cpu().numpy()
    rot = result[:,-2,:]
    trans = result[:,-1,:]
    res_mesh = result[:,:-2,:]
    rec_verts = np.matmul(evecs, res_mesh*std+mean)

    rot_mat = R.from_rotvec(rot).as_matrix()
    for j in range(90):
        _ = trimesh.Trimesh(np.matmul(rot_mat[j], rec_verts[j].T).T+trans[j], smpl_faces).export("render/test/t"+str(j)+".obj")


import matplotlib.pyplot as plt
classes = ["walk", "arm", "jump", "run"]
fig = plt.figure()
for x in range(2):
    for y in range(4):
        ax = plt.subplot(2, 4, x*4+y+1)
        ax.set_title(classes[model_kwargs['y']['action'][x*4+y]])
        ax.plot(x_start[:,:,:-2,:][x*4+y][:,:5,0].cpu().numpy())
plt.show()