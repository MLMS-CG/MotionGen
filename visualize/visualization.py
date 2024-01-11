# recover human mesh from coefs and visualize the motion

import trimesh
import mmap
import torch
import json
import numpy as np
import pyrender
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

def animate_imgs(seq_imgs):
    frames = []
    fig = plt.figure()
    for i in seq_imgs:
        frames.append([plt.imshow(i, animated=True)])
    ani= animation.ArtistAnimation(fig, frames, interval=int(1000/30), blit=True,
                                repeat_delay=0)
    plt.show()

    

def seq2imgs(meshes):
    seq_imgs = []

    for mesh in meshes:
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        # compose scene
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
        camera = pyrender.PerspectiveCamera( yfov=np.pi / 4.0)
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

        scene.add(pmesh, pose=np.eye(4))
        scene.add(light, pose=np.eye(4))

        c = 2**-0.5
        scene.add(camera, pose=[[ 1,  0,  0,  0],
                                [ 0,  c, -c, -2],
                                [ 0,  c,  c,  2],
                                [ 0,  0,  0,  1]])

        # render scene
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(scene)
        seq_imgs.append(color)
    return seq_imgs
        

def demo():
    # load a sequence from train dataset and visualize it
    # load length data
    filename_lengths = opt["path_dataset"] + "lengths.bin"
    with open(filename_lengths, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        lengths = np.frombuffer(mm, dtype=int)
    # accumulate lenght
    sum_lengths = np.array(
        [np.sum(lengths[:i]) for i in range(len(lengths))]
    )

    # load coefs data
    filename_dataset = opt["path_dataset"] + "dataset.bin"
    mm_dataset = np.memmap(
            filename_dataset, dtype="float32", mode="r"
        )

    # build indices
    nb_freqs = opt["nb_freqs"]
    chunkIndexStartFrame = []
    for i in range(len(lengths)):
        current_length = lengths[i]
        start_frame = sum_lengths[i] * nb_freqs * 3
        end_frame = current_length * nb_freqs * 3
        chunkIndexStartFrame.append([i, start_frame, end_frame])

    # reproduce the mesh by coefs and evecs
    sequence = mm_dataset[chunkIndexStartFrame[0][1]:chunkIndexStartFrame[0][2]]
    sequence = sequence.reshape(-1, nb_freqs, 3)
    meshes = np.matmul(evecs.cpu().numpy(), sequence)
    meshes = [trimesh.Trimesh(mesh, smpl_faces) for mesh in meshes]

    seq_imgs = seq2imgs(meshes)
    animate_imgs(seq_imgs) 
    input("Press Enter to continue...")

    # visualize the whole sequence
    # verts = np.zeros((0, 3))
    # faces = np.zeros((0, 3))
    # for i in range(0,len(meshes), 5):
    #     faces = np.concatenate((faces, smpl_faces + len(verts)))
    #     verts = np.concatenate((verts, meshes[i]))

    # mesh = trimesh.Trimesh(verts, faces)
    # mesh.show()



    # TODO: visualize the sequence frame by frame

if __name__ == "__main__":
    demo()
