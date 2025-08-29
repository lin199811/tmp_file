#!/bin/bash

NUM_GPUs=${1:-2}
PLATFORM=quadruped
N=2

CFG_FILE=./cfgs/DA/phase${N}_vehicle_${PLATFORM}/st3d/pvrcnn_st3d.yaml

CKPT=/home/uqhyan14/vlm_challenge/track5/output/cfgs/DA/phase${N}_vehicle_${PLATFORM}/st3d/pvrcnn_st3d/default/ckpt/checkpoint_epoch_2.pth

CKPT=/home/uqhyan14/vlm_challenge/track5/data/track5-cross-platform-3d-object-detection/pretrained/phase2/pvrcnn_st3d.pth


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