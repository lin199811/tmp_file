#!/bin/bash

NUM_GPUs=${1:-2}
PLATFORM=quadruped
N=2

CFG_FILE=./cfgs/DA/phase${N}_vehicle_${PLATFORM}/source_only/pvrcnn_source.yaml

CFG_FILE=./cfgs/DA/phase${N}_vehicle_${PLATFORM}/source_only/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml
# CFG_FILE=./cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml
PRETRAINED_MODEL=/scratch/project_mnt/S0202/vlm_challenge/LION/ckpt/checkpoint_epoch_36_nus_mamba.pth

if [ -n "${PRETRAINED_MODEL}" ]; then
    CKPT_ARG="--pretrained_model ${PRETRAINED_MODEL}"
else
    CKPT_ARG=" "
fi

if [ ${NUM_GPUs} -gt 1 ]; then
    echo "Running with ${NUM_GPUs} GPUs..."
    bash scripts/dist_train.sh ${NUM_GPUs} \
        --cfg_file ${CFG_FILE} \
         ${CKPT_ARG}
else
    echo "Running with single GPU..."
    python train.py \
        --cfg_file ${CFG_FILE} \
         ${CKPT_ARG}
fi