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

    betas = np.array(gd.get_named_beta_schedule(args.noise_schedule, args.diffusion_steps, 1))
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    def _extract_into_tensor(arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def q_sample(x_start, t, noise=None):
        return (
            _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    for epoch in range(100):
        train_loss, val_loss = [],[]
        train_con_loss = []
        train_rec_loss = []
        model.train()
        for batch in train_data:
            output, target = batch
            output = output.to("cuda").unsqueeze(1)
            target = target.to("cuda").unsqueeze(1)

            mp_trainer.zero_grad()
            output = (output.to(torch.float32).to("cuda")-coef_mean)/coef_std
            t = sample(args.batch_size, max(epoch*20,1), "cuda")
            noise = torch.rand_like(output).to("cuda")
            x_t = q_sample(output, t, noise=noise)
            recon, rep = model(x_t, t)

            recon = recon.squeeze()
            rep = rep.squeeze()

            target = (target.to(torch.float32).to("cuda")-coef_mean)/coef_std
            t_t = q_sample(target, torch.zeros_like(t).to("cuda"), noise=noise)
            _, rep_ori = model(t_t, torch.zeros_like(t).to("cuda"))

            rep_ori = rep_ori.squeeze()

            loss_rec = (((recon-x_t.squeeze())**2)*coef_std).sum((-2,-1)).mean(0)
            loss_con = contrastive_loss(rep, rep_ori).mean()
            loss = loss_rec + loss_con
            train_loss.append(loss.item())
            train_con_loss.append(loss_con.item())
            train_rec_loss.append(loss_rec.item())
            mp_trainer.backward(loss)
            mp_trainer.optimize(opt)
        
        with torch.no_grad():
            model.eval()
            for batch in val_data:
                output, target = batch
                output = output.to("cuda").unsqueeze(1)
                target = target.to("cuda").unsqueeze(1)

                output = (output.to(torch.float32).to("cuda")-coef_mean)/coef_std
                target = (target.to(torch.float32).to("cuda")-coef_mean)/coef_std

                t = sample(args.batch_size, args.diffusion_steps, "cuda")
                noise = torch.rand_like(output).to("cuda")
                x_t = q_sample(output, t, noise=noise)
                recon, rep = model(x_t, t)
                recon = recon.squeeze()
                rep = rep.squeeze()
                _, rep_ori = model(target, torch.zeros_like(t).to("cuda"))
                rep_ori = rep_ori.squeeze()
                loss_rec = (((recon-x_t.squeeze())**2)*coef_std).sum((-2,-1)).mean(0)
                loss_con = contrastive_loss(rep, rep_ori).mean()
                loss = loss_rec + loss_con*10
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
        os.path.join("save/classifier/", f"model{step:04d}.pt"),
    )
    torch.save(opt.state_dict(), os.path.join("save/classifier/", f"opt{step:06d}.pt"))

if __name__ == "__main__":
    main()
