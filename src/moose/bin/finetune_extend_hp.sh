#!/bin/bash

HOME_LOC=/path/to/your/home
ENV_PATH=/path/to/your/env
SCRIPT_LOC=$HOME_LOC/TR2-D2/tr2d2-pep
LOG_LOC=$HOME_LOC/TR2-D2/tr2d2-pep/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='tr2d2_alpha_0.25_buffer_size_100_exploration_0.25'
PYTHON_EXECUTABLE=$ENV_PATH/bin/python

# ===================================================================

#source "$(conda info --base)/etc/profile.d/conda.sh"
#conda activate moose

# ===================================================================

python /scratch/pranamlab/shared/moose/src/moose/models/finetune.py \
    --device "cuda:1" \
    --noise_removal \
    --wdce_num_replicates 16 \
    --buffer_size 100 \
    --seq_length 20 \
    --num_children 50 \
    --total_num_steps 128 \
    --num_iter 10 \
    --resample_every_n_step 10 \
    --num_epochs 1000 \
    --exploration 0.25 \
    --save_every_n_epochs 50 \
    --reset_every_n_step 1 \
    --alpha 0.25 \
    --name tr2d2ft_alpha_0.25_buffer_size_100_exploration_0.25 \
    --pt_model_path /scratch/pranamlab/shared/moose/model.ckpt \
    --grad_clip > "/scratch/pranamlab/liz/penn_work/moose/logs/${DATE}_${SPECIAL_PREFIX}.log" 2>&1

#conda deactivate