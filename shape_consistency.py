import sys
sys.path.insert(0, "/home/kxue/Work/MotionGen/MotionGen/")

import os
import json
from utils import dist_util
import torch
from utils.fixseed import fixseed
from utils.parser_util import train_args
from model.classifier import Classifier
import numpy as np
import mmap


def main():
    args = train_args()
    fixseed(args.seed)

    args.overwrite = True
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and (not args.overwrite):
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    coef_mean = torch.tensor(np.load("data/classifier/coef_mean.npy")).to("cuda")
    coef_std = torch.tensor(np.load("data/classifier/coef_std.npy")).to("cuda")
    model = Classifier().to("cuda")
    model.load_state_dict(dist_util.load_state_dict(
        "save/autoencoder/model0099.pt", map_location=dist_util.dev()
    ))
    model.eval()




if __name__ == "__main__":
    main()