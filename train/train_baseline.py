import sys
sys.path.insert(0, "/home/kxue/Work/MotionGen/MotionGen/")

import os
import json
from utils import dist_util
import torch
from utils.fixseed import fixseed
from utils.parser_util import train_args
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_unconditioned_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform 

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')
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

    print("creating data loader...")

    train_data, means_stds = get_dataset_loader(
        "train", args.data_dir, args.batch_size, args.nb_freqs, args.offset, args.size_window, return_gender=args.return_gender, rot_aug=args.rot_aug
    )
    # val_data, _ = get_dataset_loader(
    #     "val",  args.data_dir, args.batch_size, args.nb_freqs, args.offset, args.size_window, means_stds, return_gender=args.return_gender, rot_aug=args.rot_aug
    # )

    if args.cuda:
        means_stds = [torch.tensor(ele).to("cuda") for ele in means_stds]

    print("creating model and diffusion...")
    model, diffusion = create_unconditioned_model_and_diffusion(args, means_stds)
    model.tpose_ae.load_state_dict(dist_util.load_state_dict(
        "save/autoencoder/model0099.pt", map_location=dist_util.dev()
    ))
    model.tpose_ae.requires_grad_(False)
    if args.cuda:
        model.to("cuda")

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, [train_data, None]).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
