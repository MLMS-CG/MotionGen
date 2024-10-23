#!/bin/sh
cd /home/kxue/Work/MotionGen/MotionGen

# python train/train_baseline.py --save_dir save/walk2000_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_multiclass/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 256 --batch_size 64 --used_id 0

# python train/train_baseline.py --save_dir save/throw2000_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_multiclass/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 256 --batch_size 64 --used_id 5

# python train/train_baseline.py --save_dir save/kick2000_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_multiclass/ --train_platform_type ClearmlPlatform --num_steps 40_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 256 --batch_size 64 --used_id 6

# python train/train_baseline.py --save_dir save/jump2000_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_multiclass/ --train_platform_type ClearmlPlatform --num_steps 40_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 256 --batch_size 64 --used_id 1

# python train/train_baseline.py --save_dir save/run2000_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_multiclass/ --train_platform_type ClearmlPlatform --num_steps 40_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 256 --batch_size 64 --used_id 2

# python train/train_baseline.py --save_dir save/stretch2000_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_multiclass/ --train_platform_type ClearmlPlatform --num_steps 40_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 256 --batch_size 64 --used_id 4

# python train/train_baseline.py --save_dir save/gesture2000_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_multiclass/ --train_platform_type ClearmlPlatform --num_steps 40_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 256 --batch_size 64 --used_id 7

# python train/train_baseline.py --save_dir save/allclasses2000_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_multiclass/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 256 --batch_size 64

# python train/train_baseline.py --save_dir save/sit2000_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_multiclass/ --train_platform_type ClearmlPlatform --num_steps 40_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 256 --batch_size 64 --used_id 3

# python train/train_baseline.py --save_dir save/jump_dyna_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dyna/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 256 --batch_size 64 --used_id 1

# python train/train_baseline.py --save_dir save/walk_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/beta/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 1 --lambda_rot 1 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 128 --batch_size 128 --used_id 0

# python train/train_baseline.py --save_dir save/crawl_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/betacrawl/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 1 --lambda_rot 1 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 128 --batch_size 128 --used_id 8

# python train/train_baseline.py --save_dir save/jump_dmplbeta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/betajump_dyna/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 1 --lambda_rot 1 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 128 --batch_size 128 --used_id 1

# python train/train_baseline.py --save_dir save/jump_dmpl_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/meshDynaJump/ --train_platform_type ClearmlPlatform --num_steps 100_000 --lambda_trans 1 --lambda_rot 1 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 128 --batch_size 64 --used_id 1

# python train/train_baseline.py --save_dir save/walk_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/betawalk/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 1 --lambda_rot 1 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 128 --batch_size 128 --used_id 0

# python train/train_baseline.py --save_dir save/jump_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/betajump/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 1 --lambda_rot 1 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 128 --batch_size 128 --used_id 1

# python train/train_baseline.py --save_dir save/kick_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/betakick/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 1 --lambda_rot 1 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 128 --batch_size 128 --used_id 6

# python train/train_baseline.py --save_dir save/run_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/betarun/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 1 --lambda_rot 1 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 128 --batch_size 128 --used_id 2

# python train/train_baseline.py --save_dir save/throw_beta_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/betathrow/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 1 --lambda_rot 1 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 128 --batch_size 128 --used_id 5

python train/train_baseline.py --save_dir save/all_beta --data_dir data/datasets/beta_all/ --train_platform_type ClearmlPlatform --num_steps 50_000 --lambda_trans 1 --lambda_rot 1 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 128 --batch_size 128