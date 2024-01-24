#!/bin/sh
cd /home/kxue/Work/MotionGen/MotionGen

'''
# 17/07/2024
python train/train_baseline.py --save_dir save/epsilon_linear_mesh1_velo1 --lambda_mesh_mse 1 --lambda_mesh_velo 1 --noise_schedule linear
python train/train_baseline.py --save_dir save/epsilon_linear_mesh10_velo10 --lambda_mesh_mse 10 --lambda_mesh_velo 10 --noise_schedule linear
python train/train_baseline.py --save_dir save/epsilon_linear_mesh1_velo10 --lambda_mesh_mse 1 --lambda_mesh_velo 10 --noise_schedule linear
python train/train_baseline.py --save_dir save/epsilon_linear_mesh10_velo1 --lambda_mesh_mse 10 --lambda_mesh_velo 1 --noise_schedule linear
# python train/train_baseline.py --save_dir save/linear_mesh0_velo1 --lambda_mesh_mse 0 --lambda_mesh_velo 1 --noise_schedule linear
# python train/train_baseline.py --save_dir save/linear_mesh1_velo0 --lambda_mesh_mse 1 --lambda_mesh_velo 0 --noise_schedule linear
python train/train_baseline.py --save_dir save/epsilon_cosine_mesh1_velo1 --lambda_mesh_mse 1 --lambda_mesh_velo 1 --noise_schedule cosine
python train/train_baseline.py --save_dir save/epsilon_cosine_mesh10_velo10 --lambda_mesh_mse 10 --lambda_mesh_velo 10 --noise_schedule cosine
python train/train_baseline.py --save_dir save/epsilon_cosine_mesh1_velo10 --lambda_mesh_mse 1 --lambda_mesh_velo 10 --noise_schedule cosine
python train/train_baseline.py --save_dir save/epsilon_cosine_mesh10_velo1 --lambda_mesh_mse 10 --lambda_mesh_velo 1 --noise_schedule cosine
# python train/train_baseline.py --save_dir save/cosine_mesh0_velo1 --lambda_mesh_mse 0 --lambda_mesh_velo 1 --noise_schedule cosine
# python train/train_baseline.py --save_dir save/cosine_mesh1_velo0 --lambda_mesh_mse 1 --lambda_mesh_velo 0 --noise_schedule cosine
'''

# 18/01/2024
# python train/train_baseline.py --save_dir save/x0_linear_mesh1_velo1 --lambda_mesh_mse 1 --lambda_mesh_velo 1 --noise_schedule linear --target x0
# python train/train_baseline.py --save_dir save/x0_linear_mesh1_velo0 --lambda_mesh_mse 1 --lambda_mesh_velo 0 --noise_schedule linear --target x0

# 19/01/2024
# python train/train_baseline.py --save_dir save/x0_sigma_linear_mesh1_velo1 --lambda_mesh_mse 1 --lambda_mesh_velo 1 --noise_schedule linear --target x0 --learn_sigma --batch_size 40

# 22/01/2024
# python train/train_baseline.py --save_dir save/gender_x0_linear_mesh1_velo1 --lambda_mesh_mse 1 --lambda_mesh_velo 1 --noise_schedule linear --target x0 --return_gender

# 23/01/2024 
python train/train_baseline.py --save_dir save/newdata_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/
python train/train_baseline.py --save_dir save/newdata_gender_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024/ --return_gender
python train/train_baseline.py --save_dir save/rootdata_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_root/
python train/train_baseline.py --save_dir save/rootdata_gender_x0_linear_mesh1_velo1 --data_dir data/datasets/dataset_MI_1024_root/ --return_gender