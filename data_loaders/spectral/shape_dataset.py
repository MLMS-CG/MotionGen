from torch.utils import data
import numpy as np
import mmap
import types
from scipy.spatial.transform import Rotation as R
import random
import torch
from human_body_prior.body_model.body_model import BodyModel
import json

dataPath = "/home/kxue/Work/MotionGen/HumanML3D/"
with open("preProcessing/default_options_dataset.json", "r") as outfile:
    opt = json.load(outfile)

# evecs
print("loading evecs")
with open("data/evecs_4096.bin", "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    evecs = torch.tensor(np.frombuffer(
        mm[:], dtype=np.float32)).view(6890, 4096).to(opt["device"])

evecs = evecs[:, :opt["nb_freqs"]].transpose(0, 1)

# Load body model
print("loading body models")
num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

# male
bm_male = "data/smplh/male/model.npz"
dmpl_male = "data/dmpls/male/model.npz"

smplh_data = np.load(bm_male)
dmpl_data = np.load(dmpl_male)

print("loading male")
bm_male = BodyModel(
    bm_fname=bm_male,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=dmpl_male,
)

print("sending to gpu")
bm_male = bm_male.to(opt["device"])

# female
print("loading female")
bm_female = "data/smplh/female/model.npz"
dmpl_female = "data/dmpls/female/model.npz"

bm_female = BodyModel(
    bm_fname=bm_female,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=dmpl_female,
)


class ShapeSpec(data.Dataset):
    def __init__(self, ):

        fm_beta_mean_std = np.load("data/classifier/fm_beta_mean_std.npy")
        self.fmean, self.fstd, self.mmean, self.mstd = fm_beta_mean_std
        self.len_pose_female = np.load("data/classifier/len_pose_female.npy")
        self.len_pose_male = np.load("data/classifier/len_pose_male.npy")
        self.pose_female = np.load("data/classifier/pose_female.npy")
        self.pose_male = np.load("data/classifier/pose_male.npy")

        self.len = 1e5

        self.seq_pose_male = len(self.pose_male)
        self.seq_pose_female = len(self.pose_female)

        self.gender = torch.randint(self.len,(0,1))
        self.seq_female = torch.randint(self.len,(0,self.seq_pose_female))
        self.seq_male = torch.randint(self.len,(0,self.seq_pose_male))
        
        self.frame_female = torch.zeros_like(self.gender)
        self.frame_male = torch.zeros_like(self.gender)

        for i in range(self.len):
            self.frame_female[i] = np.random.randint(self.len_pose_female[self.seq_female[i]])
            self.frame_male[i] = np.random.randint(self.len_pose_male[self.seq_male[i]])

    def __getitem__(self, idx):
        if self.gender[idx]==0:
            # male
            betas = torch.normal(self.mmean, self.mstd)
            poseFile = self.seq_male[idx]
            frame = self.frame_male[idx]
            current_bm = bm_male
        else:
            # female
            betas = torch.normal(self.fmean, self.fstd)
            poseFile = self.seq_female[idx]
            frame = self.frame_female[idx]
            current_bm = bm_female
        data = np.load(dataPath + poseFile)
        body_parms = {
            "pose_body": torch.Tensor(
                data.f["poses"][frame, 3:66]
            ).to(opt["device"]),
            # controls the finger articulation
            "pose_hand": torch.Tensor(
                data.f["poses"][frame, 66:]
            ).to(opt["device"]),
            # controls the body shape
            "betas": torch.Tensor(
                betas
            ).to(opt["device"]),
        }
        body_pose_beta = current_bm(**body_parms)
        roots = body_pose_beta.Jtr[0]
        r1 = R.from_rotvec([0.5*np.pi,0,0]).as_matrix()
        vert = np.matmul(
            r1,
            (body_pose_beta.v-roots).cpu().numpy().T
        ).T
        return vert

    def __len__(self):
        return self.len

