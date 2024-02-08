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
python train/train_baseline.py --save_dir save/pre_rerot10_trans20_resT1e4_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_res_trans 1e4 --lambda_trans 20 --lambda_rot 10

# TODO
# python train/train_baseline.py --save_dir save/pre_rerot_trans_resT1e5_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_sv_rerot_trans/ --train_platform_type ClearmlPlatform --num_steps 70_000 --lambda_res_trans 1e5
