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