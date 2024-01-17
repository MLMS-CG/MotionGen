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

args = train_args()

def get_result_iter(model, diffusion, data, t):
    t = torch.tensor(t).to("cuda")
    data = torch.tensor(data).to("cuda").unsqueeze(0)
    noise = torch.randn_like(data)
    with torch.no_grad():
        while torch.all(t>=0):
            data_t = diffusion.q_sample(data, t, noise)
            data = model(data_t, t)
            t -= 1

    return data[0].to("cpu").detach().numpy()

def get_result(model, diffusion, data, t):
    t = torch.tensor(t).to("cuda")
    data = torch.tensor(data).to("cuda").unsqueeze(0)
    noise = torch.randn_like(data)

    data_t = diffusion.q_sample(data, t, noise)
    data = model(data_t, t)

    return data[0].to("cpu").detach().numpy()


def result2mesh(data):
    meshes = np.matmul(evecs.cpu().numpy(), data)
    return meshes


def training_perform():
    train_data = get_dataset("train", args.data_dir, args.nb_freqs, args.offset, args.size_window, None)
    means_stds = train_data.means_stds
    val = get_dataset("val", args.data_dir, args.nb_freqs, 1, args.size_window, means_stds)

    means_stds = [torch.tensor(ele) for ele in means_stds]
    if args.cuda:
        means_stds = [ele.to("cuda") for ele in means_stds]
    model, diffusion = create_unconditioned_model_and_diffusion(args, means_stds)
    if args.cuda:
        model.to("cuda")
    # load checkpoints
    model.load_state_dict(
        dist_util.load_state_dict(
            "./save/unconditioned_concat_x0/model000030000.pt", map_location=dist_util.dev()
        )
    )
    model.eval()

    data = train_data.__getitem__(20)
    result = get_result(model, diffusion, data, [1999])
    mean, std = train_data.means_stds
    ori_verts = np.matmul(evecs.cpu().numpy(), data*std+mean)
    rec_verts = np.matmul(evecs.cpu().numpy(), result*std+mean)
    ori_meshes = [trimesh.Trimesh(mesh, smpl_faces) for mesh in ori_verts]
    rec_meshes = [trimesh.Trimesh(mesh, smpl_faces) for mesh in rec_verts]
    seq_imgs = seq2imgs(rec_meshes)
    frames = []
    fig = plt.figure()
    for i in seq_imgs:
        frames.append([plt.imshow(i, animated=True)])
    ani= animation.ArtistAnimation(fig, frames, interval=int(1000/30), blit=True,
                                repeat_delay=0)
    plt.show()

if __name__ == "__main__":
    training_perform()