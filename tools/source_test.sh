#!/bin/bash

NUM_GPUs=${1:-2}

PLATFORM=quadruped
N=2

CFG_FILE=./cfgs/DA/phase${N}_vehicle_${PLATFORM}/source_only/pvrcnn_st3d.yaml

CFG_FILE=./cfgs/DA/phase${N}_vehicle_${PLATFORM}/source_only/lion_mamba_nusc_8x_1f_1x_one_stride_128dim_test.yaml
CFG_FILE=./cfgs/DA/phase${N}_vehicle_${PLATFORM}/st3d/lion_mamba_nusc_8x_1f_1x_one_stride_128dim_test.yaml
# CFG_FILE=./cfgs/DA/phase${N}_vehicle_${PLATFORM}/st3d/lion_retnet_nusc_8x_1f_1x_one_stride_128dim.yaml

# CFG_FILE=./cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml
# CKPT=/home/uqhyan14/vlm_challenge/track5/data/track5-cross-platform-3d-object-detection/pretrained/phase2/pvrcnn_source.pth

CKPT=/home/uqhyan14/vlm_challenge/track5/output/cfgs/DA/phase${N}_vehicle_${PLATFORM}/source_only/pvrcnn_source/default/ckpt/checkpoint_epoch_6.pth
CKPT=/scratch/project_mnt/S0202/vlm_challenge/LION/ckpt/checkpoint_epoch_36_nus_mamba.pth
# CKPT=/home/uqhyan14/vlm_challenge/LION/tools/ckpt/checkpoint_epoch_36_nus_retnet.pth

# ----------------------
# 运行逻辑
# ----------------------

if [ -n "${CKPT}" ]; then
    CKPT_ARG="--ckpt ${CKPT}"
else
    CKPT_ARG="--eval_all"
fi

if [ ${NUM_GPUs} -gt 1 ]; then
    echo "Running with ${NUM_GPUs} GPUs..."
    bash scripts/dist_test.sh ${NUM_GPUs} \
        --cfg_file ${CFG_FILE} \
        --batch_size 16 \
        ${CKPT_ARG}
else
    echo "Running with single GPU..."
    python test.py \
        --cfg_file ${CFG_FILE} \
        --batch_size 16 \
        ${CKPT_ARG}
fi