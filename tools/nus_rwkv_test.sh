#!/bin/bash

NUM_GPUs=${1:-2}

PLATFORM=quadruped
N=2

CFG_FILE=./cfgs/DA/phase${N}_vehicle_${PLATFORM}/source_only/lion_rwkv_nusc_8x_1f_1x_one_stride_128dim_test.yaml

CKPT=/home/uqhyan14/vlm_challenge/LION/tools/ckpt/checkpoint_epoch_36_nus_retnet.pth
CKPT=/home/uqhyan14/vlm_challenge/LION/output/cfgs/DA/phase2_vehicle_quadruped/source_only/lion_rwkv_nusc_8x_1f_1x_one_stride_128dim/default/ckpt/checkpoint_epoch_7.pth


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