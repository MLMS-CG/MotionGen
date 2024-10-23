# generate motion from noise
# check denoise performence

import json
import torch
import numpy as np
import mmap
from utils import dist_util
from data_loaders.get_data import get_dataset
from utils.model_util import create_unconditioned_model_and_diffusion
from scipy.spatial.transform import Rotation as R
import os
from model.cfg_sampler import ClassifierFreeSampleModel
from tqdm import tqdm
import trimesh

torch.backends.cudnn.enabled = False
exp_name = "jump_dmpl_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1/"
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
args.batch_size = 4
scaler = 0.5

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
    train_data = get_dataset("train", args.data_dir, args.nb_freqs, args.offset, args.size_window, None, used_id=1)
    means_stds = train_data.means_stds

    means_stds = [torch.tensor(ele) for ele in means_stds]
    if args.cuda:
        means_stds = [ele.to("cuda") for ele in means_stds]
    model, diffusion = create_unconditioned_model_and_diffusion(args, means_stds)
    model = ClassifierFreeSampleModel(model)

    if args.cuda:
        model.to("cuda")
    # load checkpoints
    model.model.load_state_dict(
        dist_util.load_state_dict(
            path + "model000100000.pt", map_location=dist_util.dev()
        )
    )
    model.eval()
    mean, std = train_data.means_stds

    # tposes = np.load("data/datasets/dataset_MI_1024_sv_walk_arm_jump_run/"+"target.npy")
    tposes = [np.load("jump_mesh_tpose.npy")]
    # tpose = trimesh.load("cape.obj")
    target = torch.tensor((tposes[0].astype(np.float32)-mean)/std).to("cuda")
    # target = torch.tensor((np.matmul(evecs.cpu().numpy().T, tpose.vertices)-mean)/std).to("cuda").float()

    # _ = trimesh.Trimesh(np.matmul(evecs.cpu().numpy(), target.cpu().numpy()*std+mean), smpl_faces).export("tpose.obj")

    target = target[(None,)*2].repeat(args.batch_size,1,1,1)
    
    # actions = ["walk", "arm", "jump", "run"]
    # actions = ["walk", "jump", "run", "sit", "stretch", "throw", "kick", "gesture"]

    # generated_result = []

    action = np.random.choice(np.arange(1),args.batch_size)
    actioncond = [1 for i in range(args.batch_size)]
    cond = {'y': {'tpose': target}}
    cond['y'].update({'action': torch.tensor(action).unsqueeze(1)})
    cond['y'].update({'actioncond': torch.tensor(actioncond).to("cuda")})
    cond['y'].update({'shapecond': torch.ones_like(cond['y']['actioncond'])})

    # original generation process 
    result = get_result(model, diffusion, (args.batch_size,90,1026,3), model_kwargs=cond)

    rot = result[:,:,-2,:]
    trans = result[:,:,-1,:]
    res_mesh = result[:,:,:-2,:]
    rec_verts = np.matmul(evecs.cpu().numpy(), res_mesh*std+mean)

    save_path = "render/shaped_"+exp_name
    os.makedirs(save_path, exist_ok=True)
    motion = []
    for i in range(args.batch_size):
        rot_mat = R.from_rotvec(rot[i]).as_matrix()
        for f in range(90):
            motion.append(np.matmul(rot_mat[f], rec_verts[i,f].T).T + trans[i,f])
    # motion = np.stack(motion)
    # np.save("save/generated_result/meshes_"+str(ff)+".npy", motion)
    # np.save("save/generated_result/tpose_"+str(ff)+".npy", targetindex)
    # np.save("save/generated_result/act_"+str(ff)+".npy", action)
            # generated_result.append(motion)
    for k in range(len(motion)):
        _ = trimesh.Trimesh(motion[k], smpl_faces).export(save_path+"/1_"+str(k)+".obj")
    # generated_result = np.stack(generated_result)
    
    # penetration


    # float


if __name__ == "__main__":
    training_perform()


