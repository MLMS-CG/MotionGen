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
import trimesh

# each exp correspond to different settings
exps = [
    ["jump_dmplbeta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1/"],
    ["throw_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1/"],
    ["kick_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1/"],
    ["run_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1/"],
    ["jump_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1/"],
    ["jack_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1/"],
    ["crawl_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1/"],
    ["walk_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1/"],
]

torch.backends.cudnn.enabled = False
exp_name = exps[0]
path = "./save/" + exp_name

with open("preProcessing/default_options_dataset.json", "r") as outfile:
    opt = json.load(outfile)


path_faces = "data/faces.bin"

r1 = R.from_rotvec([0.5*np.pi,0,0]).as_matrix()

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
args.batch_size = 1
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


def training_perform():
    train_data = get_dataset("train", args.data_dir, args.nb_freqs, args.offset, args.size_window, None, used_id=1)
    means_stds = train_data.means_stds

    means_stds = [torch.tensor(ele) for ele in means_stds]
    if args.cuda:
        means_stds = [ele.to("cuda") for ele in means_stds]
    model, diffusion = create_unconditioned_model_and_diffusion(args, means_stds)
    model.mode('beta')
    model = ClassifierFreeSampleModel(model)

    if args.cuda:
        model.to("cuda")
    # load checkpoints
    model.model.load_state_dict(
        dist_util.load_state_dict(
            path + "model000030000.pt", map_location=dist_util.dev()
        )
    )
    model.eval()
    mean, std = train_data.means_stds 

    # target mesh
    # targetBeta = np.load("id25.npy")
    targetBeta = np.load("targetMeshes/beta5.npy")
    # targetBeta = np.load("tposes/betas_male.npy")[221]

    # load mean and std used for training
    meanBeta = np.load("beta_all_mean.npy")
    stdBeta = np.load("beta_all_std.npy")
    target = torch.tensor((targetBeta-meanBeta)/stdBeta).float().to("cuda")

    # _ = trimesh.Trimesh(np.matmul(evecs.cpu().numpy(), target.cpu().numpy()*std+mean), smpl_faces).export("tpose.obj")

    target = target[(None,)*2].repeat(args.batch_size,1,1)
    
    action = np.random.choice(np.arange(8,9),args.batch_size)
    actioncond = [1 for i in range(args.batch_size)]
    cond = {'y': {'tpose': target}}
    cond['y'].update({'action': torch.tensor(action).unsqueeze(1)})
    cond['y'].update({'actioncond': torch.tensor(actioncond).to("cuda")})
    cond['y'].update({'shapecond': torch.ones_like(cond['y']['actioncond'])})

    # original generation process 
    result = get_result(model, diffusion, (args.batch_size,90,77), model_kwargs=cond)

    pose = result[:,:,:63]
    rot = result[:,:,71:74]
    trans = result[:,:,74:77]
    dmpl = result[:,:,63:71]

    from human_body_prior.body_model.body_model import BodyModel
    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    # male
    bm_male = "data/smplh/male/model.npz"
    dmpl_male = "data/dmpls/male/model.npz"

    print("loading male")
    bm_male = BodyModel(
        bm_fname=bm_male,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_male,
    )

    print("sending to gpu")
    bm_male = bm_male.to("cuda")

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
    bm_female = bm_female.to("cuda")


    save_path = "render/shaped_"+exp_name
    os.makedirs(save_path, exist_ok=True)
    motion = []
    for i in range(args.batch_size):
        body_parms = {
            "pose_body": torch.Tensor(
                pose[i]*std[:63]+mean[:63]
            ).to(opt["device"]),
            # uncomment this to see dynamics, only for exp jump_dmplbeta
            # "dmpls": torch.Tensor(
            #     dmpl[i]*std[63:71]+mean[63:71]
            # ).to(opt["device"]),
            # controls the body shape
            "betas": torch.Tensor(
                np.repeat(
                    targetBeta.reshape(1,16),
                    repeats=90,
                    axis=0,
                )
            ).to(opt["device"]),
        }
        body_pose_beta = bm_female(**body_parms)
        roots = body_pose_beta.Jtr[:,0]
        
        # r1 is a reverse rotation to a rotation applied during data generation, this process will let
        # the generated motion face the correct direction
        verts = np.stack([
            np.matmul(
                r1,
                (body_pose_beta.v[k]-roots[k]).cpu().numpy().T
            ).T
            for k in range(len(body_pose_beta.v))
        ])

        rot_mat = R.from_rotvec(rot[i]).as_matrix()
        rot = R.from_matrix(rot_mat).as_rotvec()
        tran = trans[i]

        # this block is used to export data for retargeting
        # data = {"smpl_poses":np.concatenate([rot,pose[0]*std[:63]+mean[:63],np.zeros((90,6))],axis=1),"smpl_trans":tran*100}
        # import pickle
        # with open("generated results/6/jump jack/skeleton.pkl","wb") as handle:
        #     pickle.dump(data,handle,protocol=pickle.HIGHEST_PROTOCOL)

        # this will export motions to a given path
        for f in range(90):
            motion.append(np.matmul(rot_mat[f], verts[f].T).T + tran[f])
        for k in range(len(motion)):
            _ = trimesh.Trimesh(motion[k], smpl_faces).export("generated results/fat woman/jump jack/mesh/id25_"+str(k)+".obj")


if __name__ == "__main__":
    training_perform()


