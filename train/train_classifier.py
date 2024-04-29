import sys
sys.path.insert(0, "/home/kxue/Work/MotionGen/MotionGen/")

import os
import json
from utils import dist_util
import torch
from utils.fixseed import fixseed
from utils.parser_util import train_args
from data_loaders.get_data import get_dataset_classifier
from model.classifier import Classifier
from diffusion.fp16_util import MixedPrecisionTrainer
from torch.optim import AdamW
import numpy as np
from diffusion import gaussian_diffusion as gd
import torch.nn.functional as F
import mmap

with open("data/evecs_4096.bin", "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    evecs = torch.tensor(np.frombuffer(
        mm[:], dtype=np.float32)).view(6890, 4096).to("cuda")
    
evecs = evecs[:, :1024]

def sample(batch_size, time_step, device):
    """
    Importance-sample timesteps for a batch.

    :param batch_size: the number of timesteps.
    :param device: the torch device to save to.
    :return: a tuple (timesteps, weights):
                - timesteps: a tensor of timestep indices.
                - weights: a tensor of weights to scale the resulting losses.
    """
    w = np.ones(time_step)
    p = w / np.sum(w)
    indices_np = np.random.choice(time_step, size=(batch_size,), p=p)
    indices = torch.from_numpy(indices_np).long().to(device)
    return indices

def main():
    args = train_args()
    fixseed(args.seed)
    args.batch_size = 256
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
        args.batch_size, "train"
    )
    val_data = get_dataset_classifier(
        args.batch_size, "val"
    )
    coef_mean = torch.tensor(np.load("data/classifier/coef_mean.npy")).to("cuda")
    coef_std = torch.tensor(np.load("data/classifier/coef_std.npy")).to("cuda")
    model = Classifier().to("cuda")

    mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False, initial_lg_loss_scale=16.0)

    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(100):
        train_loss, val_loss = [],[]
        train_con_loss = []
        train_rec_loss = []
        train_mesh_loss = []
        model.train()
        for batch in train_data:
            pose1, pose2, target = batch
            pose1 = pose1.to("cuda").unsqueeze(1)
            pose2 = pose2.to("cuda").unsqueeze(1)
            target = target.to("cuda").unsqueeze(1)

            mp_trainer.zero_grad()
            pose1 = (pose1.to(torch.float32).to("cuda")-coef_mean)/coef_std
            pose2 = (pose2.to(torch.float32).to("cuda")-coef_mean)/coef_std
            # target_mesh = torch.matmul(evecs,target.to(torch.float32).to("cuda"))
            target_coef = (target.to(torch.float32).to("cuda")-coef_mean)/coef_std

            outp1, repp1 = model(pose1)
            outp2, repp2 = model(pose2)
            outp3, repp3 = model(target_coef)

            # meshp1 = torch.matmul(evecs, outp1*coef_std+coef_mean)
            # meshp2 = torch.matmul(evecs, outp2*coef_std+coef_mean)

            loss_rec1 = (((outp1-target_coef.squeeze())**2)*coef_std).sum((-2,-1)).mean(0)
            loss_rec2 = (((outp2-target_coef.squeeze())**2)*coef_std).sum((-2,-1)).mean(0)
            loss_rec3 = (((outp3-target_coef.squeeze())**2)*coef_std).sum((-2,-1)).mean(0)

            # loss_mesh1 = ((meshp1-target_mesh.squeeze())**2).sum((-2,-1)).mean(0)
            # loss_mesh2 = ((meshp2-target_mesh.squeeze())**2).sum((-2,-1)).mean(0)

            loss_con1 = contrastive_loss(repp1.squeeze(), repp2.squeeze()).mean()
            loss_con2 = contrastive_loss(repp3.squeeze(), repp2.squeeze()).mean()
            loss_con3 = contrastive_loss(repp3.squeeze(), repp1.squeeze()).mean()
            loss = loss_rec1 + loss_rec2 + loss_rec3 \
                + loss_con1 + loss_con2 + loss_con3
            # train_mesh_loss.append(loss_mesh1.item()+loss_mesh2.item())
            train_loss.append(loss.item())
            train_con_loss.append(loss_con1.item()+loss_con2.item()+loss_con3.item())
            train_rec_loss.append(loss_rec1.item()+loss_rec2.item()+loss_rec3.item())
            mp_trainer.backward(loss)
            mp_trainer.optimize(opt)
        
        with torch.no_grad():
            model.eval()
            for batch in val_data:
                pose1, pose2, target = batch
                pose1 = pose1.to("cuda").unsqueeze(1)
                pose2 = pose2.to("cuda").unsqueeze(1)
                target = target.to("cuda").unsqueeze(1)

                mp_trainer.zero_grad()
                pose1 = (pose1.to(torch.float32).to("cuda")-coef_mean)/coef_std
                pose2 = (pose2.to(torch.float32).to("cuda")-coef_mean)/coef_std
                target_coef = (target.to(torch.float32).to("cuda")-coef_mean)/coef_std

                outp1, repp1 = model(pose1)
                outp2, repp2 = model(pose2)
                outp3, repp3 = model(target_coef)

                # meshp1 = torch.matmul(evecs, outp1*coef_std+coef_mean)
                # meshp2 = torch.matmul(evecs, outp2*coef_std+coef_mean)

                loss_rec1 = (((outp1-target_coef.squeeze())**2)*coef_std).sum((-2,-1)).mean(0)
                loss_rec2 = (((outp2-target_coef.squeeze())**2)*coef_std).sum((-2,-1)).mean(0)
                loss_rec3 = (((outp3-target_coef.squeeze())**2)*coef_std).sum((-2,-1)).mean(0)

                # loss_mesh1 = ((meshp1-target_mesh.squeeze())**2).sum((-2,-1)).mean(0)
                # loss_mesh2 = ((meshp2-target_mesh.squeeze())**2).sum((-2,-1)).mean(0)

                loss_con1 = contrastive_loss(repp1.squeeze(), repp2.squeeze()).mean()
                loss_con2 = contrastive_loss(repp3.squeeze(), repp2.squeeze()).mean()
                loss_con3 = contrastive_loss(repp3.squeeze(), repp1.squeeze()).mean()
                loss = loss_rec1 + loss_rec2 + loss_rec3 \
                    + loss_con1 + loss_con2 + loss_con3
                val_loss.append(loss.item())
        train_data.dataset.__after_epoch__()
        save_model(mp_trainer, opt, epoch)
        print("Epoch:"+str(epoch))
        print("T_loss:"+str(np.mean(train_loss))+",Con_loss:"+str(np.mean(train_con_loss))+",Rec_loss"+str(np.mean(train_rec_loss)))
        # print("T_loss:"+str(np.mean(train_loss)))
        print("V_loss:"+str(np.mean(val_loss)))


def contrastive_loss(query, key):
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=-1)
    logits = query @ key.transpose(-1,-2)
    labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits/0.2, labels, reduction='none')
   

def save_model(mp_trainer, opt, step):
    torch.save(
        mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
        os.path.join("save/autoencoder/", f"model{step:04d}.pt"),
    )
    torch.save(opt.state_dict(), os.path.join("save/autoencoder/", f"opt{step:06d}.pt"))

if __name__ == "__main__":
    main()
