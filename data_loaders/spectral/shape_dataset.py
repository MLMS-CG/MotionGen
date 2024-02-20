from torch.utils import data
import numpy as np
import mmap
from scipy.spatial.transform import Rotation as R
import torch
from human_body_prior.body_model.body_model import BodyModel
import json

dataPath = "/home/kxue/Work/MotionGen/AMASS/"
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
).to(opt["device"])

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
).to(opt["device"])

class Sampler():
    def __init__():
        pass

    def __iter__(self):
        pass


class ShapeSpec(data.Dataset):
    def __init__(self, status):
        beta_male = np.load("data/classifier/shape_male.npy")
        beta_female = np.load("data/classifier/shape_female.npy")

        pose_hand_male = np.load("data/classifier/pose_hand_male.npy")
        pose_body_male = np.load("data/classifier/pose_body_male.npy")
        pose_hand_female = np.load("data/classifier/pose_hand_female.npy")
        pose_body_female = np.load("data/classifier/pose_body_female.npy")

        idx = np.arange(len(pose_hand_male))
        np.random.shuffle(idx)
        sh_pose_hand_male = pose_hand_male[idx]
        sh_pose_body_male = pose_body_male[idx]
        sh_pose_hand_female = pose_hand_female[idx]
        sh_pose_body_female = pose_body_female[idx]

        if status=="train":
            self.len = int(1000)
        else:
            self.len = int(100)

        self.beta_male = np.concatenate(
            [
                    np.random.choice(np.arange(len(beta_male)), 128, replace=False)\
                for _ in range(int(self.len))
            ]
        )
        self.beta_female = np.concatenate(
            [
                    np.random.choice(np.arange(len(beta_female)), 128, replace=False)\
                for _ in range(int(self.len))
            ]
        )

        male_data, sh_male_data = [], []
        female_data, sh_female_data = [], []

        r1 = torch.tensor(R.from_rotvec([0.5*np.pi,0,0]).as_matrix()).to(torch.float32).to("cuda")

        self.male_tpose = bm_male(
            betas=torch.Tensor(beta_male).to(opt["device"])
        )
        self.male_tpose = self.male_tpose.v - self.male_tpose.Jtr[:,0].unsqueeze(1)
        self.male_tpose = torch.matmul(
                r1.unsqueeze(0),
                (self.male_tpose).transpose(1,2)).transpose(1,2)
        self.male_tpose = torch.matmul(evecs, self.male_tpose).cpu().numpy()

        self.female_tpose = bm_female(
            betas=torch.Tensor(beta_female).to(opt["device"])
        )
        self.female_tpose = self.female_tpose.v - self.female_tpose.Jtr[:,0].unsqueeze(1)
        self.female_tpose = torch.matmul(
                r1.unsqueeze(0),
                (self.female_tpose).transpose(1,2)).transpose(1,2)
        self.female_tpose = torch.matmul(evecs, self.female_tpose).cpu().numpy()        

        for i in range(0,self.len*128,100):
            # male
            index = np.random.choice(np.arange(len(pose_hand_male)), 100)
            betas = beta_male[self.beta_male[i:i+100]]

            body_parms = {
                "pose_body": torch.Tensor(
                    pose_body_male[index]
                ).to(opt["device"]),
                # controls the finger articulation
                "pose_hand": torch.Tensor(
                    pose_hand_male[index]
                ).to(opt["device"]),
                # controls the body shape
                "betas": torch.Tensor(
                    betas
                ).to(opt["device"]),
            }
        
            body_pose_beta = bm_male(**body_parms)
            roots = body_pose_beta.Jtr[:,0].unsqueeze(1)
            
            vert = torch.matmul(
                r1.unsqueeze(0),
                (body_pose_beta.v-roots).transpose(1,2)).transpose(1,2)
            male_data.append(torch.matmul(evecs, vert).cpu().numpy())

            sh_body_parms = {
                "pose_body": torch.Tensor(
                    sh_pose_body_male[index]
                ).to(opt["device"]),
                # controls the finger articulation
                "pose_hand": torch.Tensor(
                    sh_pose_hand_male[index]
                ).to(opt["device"]),
                # controls the body shape
                "betas": torch.Tensor(
                    betas
                ).to(opt["device"]),
            }
        
            sh_body_pose_beta = bm_male(**sh_body_parms)
            sh_roots = sh_body_pose_beta.Jtr[:,0].unsqueeze(1)
            sh_vert = torch.matmul(
                r1.unsqueeze(0),
                (sh_body_pose_beta.v-sh_roots).transpose(1,2)).transpose(1,2)
            sh_male_data.append(torch.matmul(evecs, sh_vert).cpu().numpy())
        male_data = np.concatenate(male_data).reshape(self.len, 128, 1024,3)
        sh_male_data = np.concatenate(sh_male_data).reshape(self.len, 128, 1024,3)

        for i in range(0,self.len*128,100):
            # female
            index = np.random.choice(np.arange(len(pose_hand_female)), 100)
            betas = beta_female[self.beta_female[i:i+100]]

            body_parms = {
                "pose_body": torch.Tensor(
                    pose_body_female[index]
                ).to(opt["device"]),
                # controls the finger articulation
                "pose_hand": torch.Tensor(
                    pose_hand_female[index]
                ).to(opt["device"]),
                # controls the body shape
                "betas": torch.Tensor(
                    betas
                ).to(opt["device"]),
            }
        
            body_pose_beta = bm_female(**body_parms)
            roots = body_pose_beta.Jtr[:,0].unsqueeze(1)
            vert = torch.matmul(
                r1.unsqueeze(0),
                (body_pose_beta.v-roots).transpose(1,2)).transpose(1,2)
            female_data.append(torch.matmul(evecs, vert).cpu().numpy())

            sh_body_parms = {
                "pose_body": torch.Tensor(
                    sh_pose_body_female[index]
                ).to(opt["device"]),
                # controls the finger articulation
                "pose_hand": torch.Tensor(
                    sh_pose_hand_female[index]
                ).to(opt["device"]),
                # controls the body shape
                "betas": torch.Tensor(
                    betas
                ).to(opt["device"]),
            }
        
            sh_body_pose_beta = bm_female(**sh_body_parms)
            sh_roots = sh_body_pose_beta.Jtr[:,0].unsqueeze(1)
            sh_vert = torch.matmul(
                r1.unsqueeze(0),
                (sh_body_pose_beta.v-sh_roots).transpose(1,2)).transpose(1,2)
            sh_female_data.append(torch.matmul(evecs, sh_vert).cpu().numpy())

        female_data = np.concatenate(female_data).reshape(self.len, 128, 1024,3)
        sh_female_data = np.concatenate(sh_female_data).reshape(self.len, 128, 1024,3)

        self.data = np.concatenate([male_data, female_data], axis=1)
        self.sh_data = np.concatenate([sh_male_data, sh_female_data], axis=1)

        self.index = np.arange(self.len)

    def __getitem__(self, idx):
        batch = self.index[idx//256]
        real_batch = idx//256
        if (idx-real_batch*256)//128==0:
            tpose = self.male_tpose[self.beta_male[batch*128+(idx-real_batch*256)%128]]
        else:
            tpose = self.female_tpose[self.beta_female[batch*128+(idx-real_batch*256)%128]]
        return self.data[batch, idx%256], self.sh_data[batch, idx%256], tpose

    def __after_epoch__(self):
        np.random.shuffle(self.index)

    def __len__(self):
        return self.len*256

