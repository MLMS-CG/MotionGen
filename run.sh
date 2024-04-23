#!/bin/sh
cd /home/kxue/Work/MotionGen/MotionGen


# 18/01/2024
# python train/train_baseline.py --save_dir save/x0_linear_mesh1_velo1 --lambda_mesh_mse 1 --lambda_mesh_velo 1 --noise_schedule linear --target x0
# python train/train_baseline.py --save_dir save/x0_linear_mesh1_velo0 --lambda_mesh_mse 1 --lambda_mesh_velo 0 --noise_schedule linear --target x0

# 19/01/2024
# python train/train_baseline.py --save_dir save/x0_sigma_linear_mesh1_velo1 --lambda_mesh_mse 1 --lambda_mesh_velo 1 --noise_schedule linear --target x0 --learn_sigma --batch_size 40

# 22/01/2024
# python train/train_baseline.py --save_dir save/gender_x0_linear_mesh1_velo1 --lambda_mesh_mse 1 --lambda_mesh_velo 1 --noise_schedule linear --target x0 --return_gender

# 23/01/2024 
# python train/train_baseline.py --save_dir save/newdata_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/
# python train/train_baseline.py --save_dir save/newdata_gender_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/ --return_gender
# python train/train_baseline.py --save_dir save/rootdata_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_root/
# python train/train_baseline.py --save_dir save/rootdata_gender_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_root/ --return_gender

# 24/01/2024
# python train/train_baseline.py --save_dir save/rootdata_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_root/ --train_platform_type ClearmlPlatform
# python train/train_baseline.py --save_dir save/rootdata_gender_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_root/ --return_gender --train_platform_type ClearmlPlatform
# python train/train_baseline.py --save_dir save/shape1_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/ --train_platform_type ClearmlPlatform --lambda_shape 1 --num_steps 70_000 --shape_rep
# python train/train_baseline.py --save_dir save/shape5_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/ --train_platform_type ClearmlPlatform --lambda_shape 5 --num_steps 70_000 --shape_rep
# python train/train_baseline.py --save_dir save/shape10_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/ --train_platform_type ClearmlPlatform --lambda_shape 10 --num_steps 70_000 --shape_rep

# 25/01/2024
# python train/train_baseline.py --save_dir save/t_shape1_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/ --train_platform_type ClearmlPlatform --lambda_shape 1 --num_steps 70_000 --t_emb add --shape_rep
# python train/train_baseline.py --save_dir save/add_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/ --train_platform_type ClearmlPlatform --num_steps 60_000 --t_emb add
# python train/train_baseline.py --save_dir save/concat_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/ --train_platform_type ClearmlPlatform --num_steps 60_000
# python train/train_baseline.py --save_dir save/t_shape5_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/ --train_platform_type ClearmlPlatform --lambda_shape 5 --num_steps 60_000 --t_emb add --shape_rep
# python train/train_baseline.py --save_dir save/t_shape10_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/ --train_platform_type ClearmlPlatform --lambda_shape 10 --num_steps 60_000 --t_emb add --shape_rep

# 05/02/2024
# python train/train_baseline.py --save_dir save/transFree_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000

# 07/02/2024
# python train/train_baseline.py --save_dir save/pre_rerot_trans_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_res_trans 1e4
# python train/train_baseline.py --save_dir save/pre_rot_trans_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000
# python train/train_baseline.py --save_dir save/pre_aug_rerot_trans_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --rot_aug
# python train/train_baseline.py --save_dir save/pre_aug_rerot_trans_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --rot_aug --lambda_res_trans 1e4

# 08/02/2024
# python train/train_baseline.py --save_dir save/pre_rerot10_trans20_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_res_trans 1e4 --lambda_trans 20 --lambda_rot 10

# 09/08/2024
# python train/train_baseline.py --save_dir save/pre_rerot10_trans20_resT1e5_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_res_trans 1e5 --lambda_trans 20 --lambda_rot 10

# python train/train_baseline.py --save_dir save/pre_rerot10_trans50_resT1e5_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 80_000 --lambda_res_trans 1e5 --lambda_trans 50 --lambda_rot 10
# python train/train_baseline.py --save_dir save/pre_rerot10_trans50_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 80_000 --lambda_res_trans 1e4 --lambda_trans 50 --lambda_rot 10
# python train/train_baseline.py --save_dir save/pre_rerot10_trans100_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 80_000 --lambda_res_trans 1e4 --lambda_trans 100 --lambda_rot 10
# python train/train_baseline.py --save_dir save/pre_rerot20_trans50_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 80_000 --lambda_res_trans 1e4 --lambda_trans 50 --lambda_rot 20


# python train/train_baseline.py --save_dir save/pre_rerot20_trans50_resT0_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 0

# 16/02/2024
# python train/train_baseline.py --save_dir save/shape_rerot10_trans50_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4

# 19/02/2024
# comment baseline.py line83 to desable the residual archi in model
# python train/train_baseline.py --save_dir save/shape_rerot10_trans50_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --batch_size 48
# python train/train_baseline.py --save_dir save/shape_rand_rerot10_trans50_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 100 --lambda_rot 10 --lambda_res_trans 100
# python train/train_baseline.py --save_dir save/shape1000_rerot10_trans50_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --diffusion_steps 1000
# python train/train_baseline.py --save_dir save/shapecosine_rerot10_trans50_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine
# python train/train_baseline.py --save_dir save/shape1000cosine_rerot10_trans50_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --diffusion_steps 1000 --noise_schedule cosine

# python train/train_baseline.py --save_dir save/shapecosine_res_rerot10_trans50_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine

# python train/train_baseline.py --save_dir save/shapecosine_sty_rerot10_trans50_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine

# python train/train_baseline.py --save_dir save/walkarm_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_walk_arm/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine

# python train/train_baseline.py --save_dir save/walkarmjumprun_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_2048_sv_walk_arm_jump_run/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine

# python train/train_baseline.py --save_dir save/balance_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_walk_arm_jump_run/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 512 --batch_size 64

# python train/train_baseline.py --save_dir save/walkarm_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_walk_arm_jump_run/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 512 --batch_size 64 --diffusion_steps 1000

# python train/train_baseline.py --save_dir save/balancemirror_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_walk_arm_jump_run/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 512 --batch_size 64

# python train/train_baseline.py --save_dir save/walkarmjumprun_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_walk_arm_jump_run/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine

python train/train_baseline.py --save_dir save/multiclass_rerot10_trans50_resT1e4_x0_cosine_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_multiclass/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_trans 50 --lambda_rot 10 --lambda_res_trans 1e4 --noise_schedule cosine --latent_dim 512 --batch_size 64
