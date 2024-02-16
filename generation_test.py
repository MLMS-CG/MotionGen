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
from visualize.visualization import seq2imgs
from scipy.spatial.transform import Rotation as R
import os
from model.classifier import Classifier
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.animation as animation

torch.backends.cudnn.enabled = False
exp_name = "pre_rerot10_trans50_resT1e4_x0_linear_mesh1_velo1/"
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
    train_data = get_dataset("train", args.data_dir, args.nb_freqs, args.offset, args.size_window, None)
    means_stds = train_data.means_stds

    means_stds = [torch.tensor(ele) for ele in means_stds]
    if args.cuda:
        means_stds = [ele.to("cuda") for ele in means_stds]
    model, diffusion = create_unconditioned_model_and_diffusion(args, means_stds)
    if args.cuda:
        model.to("cuda")
    # load checkpoints
    model.load_state_dict(
        dist_util.load_state_dict(
            path + "model000070000.pt", map_location=dist_util.dev()
        )
    )
    model.eval()

    classifier = Classifier()
    classifier.load_state_dict(dist_util.load_state_dict(
        "save/autoencoder/model0050.pt", map_location=dist_util.dev()
    ))
    classifier.to("cuda")
    classifier.eval()

    y = torch.tensor(train_data.__getitem__(50)[0]).to("cuda")[:-2].unsqueeze(0).repeat(10,1,1).unsqueeze(1)
    def cond_fn(x, t):
        with torch.enable_grad():
            # if t[0]>1400:
            #     return torch.zeros_like(x).to("cuda")
            x_in = x[:,:,:-2,:].detach().requires_grad_(True)
            _, rep = classifier(x_in, t)
            _, rep_ori = classifier(y, torch.zeros_like(t).to("cuda"))
            rep = F.normalize(rep, dim=-1)
            rep_ori = F.normalize(rep_ori, dim=-1)
            logits = ((rep@rep_ori.transpose(-1,-2))-1).sum()
            grad = torch.autograd.grad(logits, x_in)[0] * 10
            return F.pad(grad,(0,0,0,2), "constant", 0)


    mean, std = train_data.means_stds

    def render_batch(res_mesh):
        rec_verts = np.matmul(evecs.cpu().numpy(), res_mesh*std+mean)
        rec_meshes = []
        for frame in range(rec_verts.shape[1]):
            shapes = rec_verts[:,frame,:,:]
            verts = []
            faces = []
            for i in np.arange(len(shapes)):
                verts.append(shapes[i]+(i*0.6,0,0))
                faces.append(smpl_faces+6890*i)
            verts = np.concatenate(verts, axis=0)
            faces = np.concatenate(faces, axis=0)
            rec_meshes.append(trimesh.Trimesh(verts, faces))
        seq_imgs = seq2imgs(rec_meshes, z_bias=2.2, x_bias=2.1, width=1200)
        return seq_imgs

    def render_single(res_mesh, index):
        rec_verts = np.matmul(evecs.cpu().numpy(), res_mesh[index]*std+mean)
        rec_meshes = [trimesh.Trimesh(mesh, smpl_faces) for mesh in rec_verts]
        seq_imgs = seq2imgs(rec_meshes)
        return seq_imgs
    
    # original generation process 
    result = get_result_classifier(model, diffusion, (args.batch_size,90,1026,3), cond_fn=cond_fn)

    rot = result[:,:,-2,:]
    trans = result[:,:,-1,:]
    res_mesh = result[:,:,:-2,:]
    rec_verts = np.matmul(evecs.cpu().numpy(), res_mesh*std+mean)

    save_path = "render/shaped_"+exp_name
    os.makedirs(save_path, exist_ok=True)
    for i in range(args.batch_size):
        rot_mat = R.from_rotvec(rot[i]).as_matrix()
        for j in range(90):
            _ = trimesh.Trimesh(np.matmul(rot_mat[j], rec_verts[i,j].T).T+trans[i,j], smpl_faces).export(save_path+"/test_ratio10"+str(i)+"_"+str(j)+".obj")

    seq_imgs = render_single(result[:,:,:-2,:],0)
    frames = []
    fig = plt.figure()
    for i in seq_imgs:
        frames.append([plt.imshow(i, animated=True)])
    ani= animation.ArtistAnimation(fig, frames, interval=int(1000/30), blit=True,
                                repeat_delay=0)
    plt.show()

if __name__ == "__main__":
    training_perform()



# verts_gen = [result[1,i,:,:] for i in range(0,90,10)]
# verts_ori = train_data.__getitem__(50)[0]
# verts_gen = [np.matmul(evecs.cpu().numpy(), verts_gen[i]*std+mean) for i in range(9)]
# verts_ori = np.matmul(evecs.cpu().numpy(), verts_ori*std+mean)
# verts = [verts_ori]
# faces = [smpl_faces]
# for i in range(9):
#     verts.append(verts_gen[i]+(0,0,0.6*(i+1)))
#     faces.append(smpl_faces+6890*(i+1))
# verts = np.concatenate(verts, axis=0)
# faces = np.concatenate(faces, axis=0)
# trimesh.Trimesh(verts, faces).show(smooth=False)