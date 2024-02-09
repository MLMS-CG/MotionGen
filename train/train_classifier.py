import sys
sys.path.insert(0, "/home/kxue/Work/MotionGen/MotionGen/")

import os
import json
from utils import dist_util
import torch
from utils.fixseed import fixseed
from utils.parser_util import train_args
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_classifier
from utils.model_util import create_unconditioned_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform 
from model.classifier import Classifier

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

    train_data = get_dataset_classifier(
        "train", args.batch_size
    )
    val_data = get_dataset_classifier(
        "val",  args.batch_size
    )

    model = Classifier()

    

if __name__ == "__main__":
    main()
