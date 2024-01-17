#!/bin/sh
cd /home/kxue/Work/MotionGen/MotionGen

python train/train_baseline.py --save_dir save/unconditioned_add_x0 --target x0 -t_emb add
python train/train_baseline.py --save_dir save/unconditioned_concat_x0 --target x0