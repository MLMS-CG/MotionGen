# generate motion from noise
# check denoise performence

import json
import torch
import numpy as np
import mmap
import trimesh
from utils import dist_util
from data_loaders.get_data import get_dataset
from utils.model_util import create_unconditioned_model_and_diffusion
from scipy.spatial.transform import Rotation as R
import os
from model.cfg_sampler import ClassifierFreeSampleModel


torch.backends.cudnn.enabled = False
exp_name = "text_conditioned/"
path = "./save/" + exp_name

with open("preProcessing/default_options_dataset.json", "r") as outfile:
    opt = json.load(outfile)

# evecs
print("loading evecs")
with open("data/evecs_4096.bin", "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    evecs = torch.tensor(np.frombuffer(
        mm[:], dtype=np.float32)).view(6890, 4096).to(opt["device"])
    
evecs = evecs[:, :opt["nb_freqs"]]

path_faces = "data/faces.bin"

with open(path_faces, "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    smpl_faces = np.frombuffer(mm[:], dtype=np.intc).reshape(-1, 3)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


with open(path + "args.json", "r") as outfile:
    args = dotdict(json.load(outfile))

args.batch_size = 10
scaler = 2 # scale factor for guidance

def get_result(model, diffusion, shape, model_kwargs=None):
    data = diffusion.p_sample_loop(model, shape, clip_denoised=False, model_kwargs=model_kwargs)
    return data.to("cpu").detach().numpy()

def get_result_classifier(model, diffusion, shape, cond_fn, model_kwargs=None):
    data = diffusion.p_sample_loop(model, shape, clip_denoised=False, cond_fn=cond_fn, model_kwargs=model_kwargs)
    return data.to("cpu").detach().numpy()

def get_x0_result_iter(model, diffusion, data, t):
    t = torch.tensor(t).to("cuda")
    data = torch.tensor(data).to("cuda").unsqueeze(0)
    noise = torch.randn_like(data)
    with torch.no_grad():
        while torch.all(t>=0):
            data_t = diffusion.q_sample(data, t, noise)
            data = model(data_t, t)
            t -= 1

    return data[0].to("cpu").detach().numpy()

def result2mesh(data):
    meshes = np.matmul(evecs.cpu().numpy(), data)
    return meshes

def training_perform():
    mean = np.load("data/datasets/text_conditioned/mean.npy")
    std = np.load("data/datasets/text_conditioned/std.npy")
    max = np.load("data/datasets/text_conditioned/max.npy")
    min = np.load("data/datasets/text_conditioned/min.npy")

    tpose_mean = torch.tensor(np.load("data/classifier/coef_mean.npy"))
    tpose_std = torch.tensor(np.load("data/classifier/coef_std.npy"))

    mean_std = torch.tensor(np.stack([mean, std]))
    mean_max = torch.tensor(np.stack([mean,np.maximum(max, -min)]))
    if args.cuda:
        mean_std = [ele.to("cuda") for ele in mean_std]
    model, diffusion = create_unconditioned_model_and_diffusion(args, mean_std)
    model = ClassifierFreeSampleModel(model, scaler)

    # model.model.tpose_ae.load_state_dict(dist_util.load_state_dict(
    #     "save/autoencoder/model0099.pt", map_location=dist_util.dev()
    # ))
    # model.tpose_ae.requires_grad_(False)

    if args.cuda:
        model.to("cuda")
    # load checkpoints
    model.model.load_my_state_dict(
        dist_util.load_state_dict(
            path + "model000100000.pt", map_location=dist_util.dev()
        )
    )
    model.eval()

    # train female_notrans_50 male_5000 male_notrans_500 male_jump_12000
    # male_walk_20000 female_return_40000 val_female_2000 val_male_11000

    with open("data/datasets/text_conditioned/Tunconditioned.bin", "r+b") as f:
        tposes = mmap.mmap(f.fileno(), 0)
    tpose = torch.frombuffer(tposes, dtype=torch.float32).view(-1,1024,3)
    target = torch.stack([(tpose[0]-tpose_mean)/tpose_std for i in range(args.batch_size)]).to("cuda")
    # target = target[(None,)*2].repeat(args.batch_size,1,1,1)

    # text conditioning signal
    text = ["a person walks forward" for i in range(args.batch_size)]
    textcond = [1 for i in range(args.batch_size)]
    lenbatch = torch.tensor([200 for i in range(args.batch_size)]).to('cuda')
    maskbatch = torch.logical_not(torch.zeros(args.batch_size, 200)).to('cuda')
    cond = {'y': {'mask': maskbatch, 'lengths': lenbatch, 'tpose': target}}
    cond['y'].update({'text': text})
    cond['y'].update({'textcond': torch.tensor(textcond).to("cuda")})
    cond['y'].update({'shapecond': torch.ones_like(cond['y']['textcond'])})

    # original generation process 
    result = get_result(model, diffusion, (args.batch_size,200,1026,3), model_kwargs=cond)

    rot = result[:,:,-2,:]
    trans = result[:,:,-1,:]
    res_mesh = result[:,:,:-2,:]
    rec_verts = np.matmul(evecs.cpu().numpy(), res_mesh*std+mean)

    save_path = "render/"+exp_name
    os.makedirs(save_path, exist_ok=True)
    for i in range(args.batch_size):
        rot_mat = R.from_rotvec(rot[i]).as_matrix()
        for j in range(200):
            _ = trimesh.Trimesh(np.matmul(rot_mat[j], rec_verts[i,j].T).T+trans[i,j], smpl_faces).export(save_path+"/test_"+str(scaler)+"_"+str(i)+"_"+str(j)+".obj")


if __name__ == "__main__":
    training_perform()

