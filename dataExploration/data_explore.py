# Check how many different person are in the AMASS dataset

## Relative packages
import numpy as np
import os

# Start from ACCAD
dataPath = "/home/kxue/Work/MotionGen/HumanML3D/"

datasets = [
    "ACCAD/ACCAD/",
    "BMLhandball/",
    "BMLmovi/BMLmovi/",
    "BMLrub/BioMotionLab_NTroje/",
    "CMU/CMU/",
    "DanceDB/DanceDB/",
    "DFaust/DFaust_67",
    "EKUT/EKUT/",
    "EyesJapanDataset/Eyes_Japan_Dataset/",
    "GRAB/",
    "HUMAN4D/HUMAN4D/",
    "HDM05/MPI_HDM05/",
    "HumanEva/HumanEva/",
    "KIT/KIT/",
    "Mosh/MPI_mosh/",
    "PosePrior/MPI_Limits/",
    "SFU/SFU/",
    "SOMA/",
    "SSM/SSM_synced/",
    "TCDHands/TCD_handMocap/",
    "TotalCapture/TotalCapture/",
    "Transitions/Transitions_mocap/",
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
            frames = data.f.poses.shape[0]
            if "mocap_framerate" in data.files:
                frameRate = data.f.mocap_framerate
            else:
                frameRate = data.f.mocap_frame_rate
            if shape not in shapes.keys():
                shapes[shape] = {"motion":[], "frames":[], "time":[], "frameRate":[]}
            shapes[shape]["motion"].append(dataset + '/' + folder + '/' + file)
            shapes[shape]["frames"].append(frames)
            shapes[shape]["frameRate"].append(frameRate)
            shapes[shape]["time"].append(frames/frameRate)

keys = list(shapes.keys())

# check if there exist a shape with different frame rate
numFrameRate = np.zeros(len(keys))
for idx, key in enumerate(keys):
    frameRate = set(np.stack(shapes[key]["frameRate"]))
    numFrameRate[idx] = len(frameRate)

np.where(numFrameRate==2)
