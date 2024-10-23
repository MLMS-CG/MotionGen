import polyscope as ps
import mmap
import numpy as np

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    'cornsilk', 'crimson', 'cyan', 'darkblue',
    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey',
    'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue'
]

path_faces = "data/faces.bin"

with open(path_faces, "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    smpl_faces = np.frombuffer(mm[:], dtype=np.intc).reshape(-1, 3)

# test the influence of using different number of eigenvectors to calculate the coefficients
# load eigenvectors
print("loading evecs")
with open("data/evecs_4096.bin", "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    evecs = np.frombuffer(
        mm[:], dtype=np.float32).reshape(6890, 4096)
    
def number_of_eignv():
    # load an original mesh
    verts = np.load("test.npy")
    # use a certain eignvetors to compress it
    used = evecs[:,:]
    # calculate the difference between the reconstructed result and the original mehs
    coefs = np.matmul(verts.T, used).T
    compressed = np.matmul(used, coefs)
    diff = np.sqrt(np.sum((compressed-verts)**2,-1))
    # visualization
    ps.init()
    mesh = ps.register_surface_mesh("recons", compressed, smpl_faces)
    mesh.set_color(tuple(int(colors[0][i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    mesh.set_smooth_shade(True)
    mesh.set_transparency(1)
    mesh.add_scalar_quantity(
        "Errors",
        diff,
        enabled=True,
        cmap='reds',
        vminmax=(0, 0.05)
    )
    ps.show()

if __name__=="__main__":
    number_of_eignv()