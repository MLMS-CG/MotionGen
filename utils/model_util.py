from model.baseline import Baseline, PredictSigma
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def create_unconditioned_model_and_diffusion(args, means_stds):
    if args.learn_sigma:
        model = PredictSigma(**get_model_args(args))
    else:
        model = Baseline(**get_model_args(args))
    diffusion = create_gaussian_diffusion(args, means_stds)
    return model, diffusion

def get_model_args(args,):
    return {'size_window': args.size_window, 't_emb': args.t_emb, 'shape':args.shape_rep,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'gender_in': args.return_gender,
            'batch_size': args.batch_size, 'n_frames':args.size_window}


def create_gaussian_diffusion(args, means_stds):
    # default params
    target = args.target 
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = args.learn_sigma
    learn_shape = args.shape_rep
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if target=="epsilon" else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        means_stds = means_stds,
        lambda_mm = args.lambda_mesh_mse,
        lambda_mv = args.lambda_mesh_velo,
        lambda_shape = args.lambda_shape,
        learn_shape = learn_shape
    )
