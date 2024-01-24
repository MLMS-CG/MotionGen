# generate motion from noise
# check denoise performence

import json
import torch
import numpy as np
import mmap
import trimesh
from utils import dist_util
from utils.parser_util import train_args
from data_loaders.get_data import get_dataset
from utils.model_util import create_unconditioned_model_and_diffusion
from visualize.visualization import seq2imgs

import matplotlib.pyplot as plt
import matplotlib.animation as animation


path = "./save/newdata_gender_x0_linear_mesh1_velo1/"

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

def get_result(model, diffusion, shape, model_kwargs=None):
    data = diffusion.p_sample_loop(model, shape, clip_denoised=False, model_kwargs=model_kwargs)
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
            path + "model000050000.pt", map_location=dist_util.dev()
        )
    )
    model.eval()

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
    
    gender = torch.ones(8, dtype=torch.int64).cuda()
    result = get_result(model, diffusion, (8,90,1024,3), {"gender":gender})
    seq_imgs = render_single(result, 0)
    frames = []
    fig = plt.figure()
    for i in seq_imgs:
        frames.append([plt.imshow(i, animated=True)])
    ani= animation.ArtistAnimation(fig, frames, interval=int(1000/30), blit=True,
                                repeat_delay=0)
    plt.show()

if __name__ == "__main__":
    training_perform()