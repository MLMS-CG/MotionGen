import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool

dataPath = "/home/kxue/Work/MotionGen/AMASS/"

fm_beta_mean_std = np.load("data/classifier/fm_beta_mean_std.npy")
fmean, fstd, mmean, mstd = fm_beta_mean_std
len_pose_female = np.load("data/classifier/len_pose_female.npy")
len_pose_male = np.load("data/classifier/len_pose_male.npy")
pose_female = np.load("data/classifier/pose_female.npy")
pose_male = np.load("data/classifier/pose_male.npy")

tl = int(5e3)

seq_pose_male = len(pose_male)
seq_pose_female = len(pose_female)

gender = torch.randint(2,(tl,))
seq_female = torch.randint(seq_pose_female, (tl,))
seq_male = torch.randint(seq_pose_male, (tl, ))

frame_female = torch.zeros_like(gender)
frame_male = torch.zeros_like(gender)

for i in range(tl):
    frame_female[i] = np.random.randint(len_pose_female[seq_female[i]])
    frame_male[i] = np.random.randint(len_pose_male[seq_male[i]])

def extract_pose(input):
    file, frame = input
    d = np.load(dataPath + file)
    if "pose_body" in d.files and "pose_hand" in d.files: 
        pose_body = d.f.pose_body[frame]
        pose_hand = d.f.pose_hand[frame]
    else:
        pose_body = d.f.poses[frame, 3:66]
        pose_hand = d.f.poses[frame,66:]
    return pose_body, pose_hand

used_argu = [(pose_male[seq_male[i]],frame_male[i] ) for i in range(tl)]

with Pool(8) as p:
    res = list(tqdm(p.imap(extract_pose, used_argu), total=tl))

np.save("data/pose_body_male.npy", np.stack([res[i][0] for i in range(tl)]))
np.save("data/pose_hand_male.npy", np.stack([res[i][1] for i in range(tl)]))

    # pose_female_file = pose_female[seq_female[i]]
    # d = np.load(dataPath + pose_female_file)
    # if "pose_body" in d.files and "pose_hand" in d.files: 
    #     pose_body_female.append(d.f.pose_body[frame_female[i]])
    #     pose_hand_female.append(d.f.pose_hand[frame_female[i]])
    # else:
    #     pose_body_female.append(d.f.poses[frame_female[i], 3:66])
    #     pose_hand_female.append(d.f.poses[frame_female[i], 66:])

used_argu = [(pose_female[seq_female[i]],frame_female[i] ) for i in range(tl)]

with Pool(8) as p:
    res = list(tqdm(p.imap(extract_pose, used_argu), total=tl))

np.save("data/pose_body_female.npy", np.stack([res[i][0] for i in range(tl)]))
np.save("data/pose_hand_female.npy", np.stack([res[i][1] for i in range(tl)]))

