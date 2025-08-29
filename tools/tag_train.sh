#!/bin/bash

NUM_GPUs=${1:-2}
PLATFORM=quadruped
N=2
# 注意：后面赋值会覆盖前面这一行
PRETRAINED_MODEL=/home/uqhyan14/vlm_challenge/track5/output/cfgs/DA/phase${N}_vehicle_${PLATFORM}/source_only/pvrcnn_source/default/ckpt/checkpoint_epoch_12.pth
PRETRAINED_MODEL=/home/uqhyan14/vlm_challenge/track5/output/cfgs/DA/phase${N}_vehicle_${PLATFORM}/source_only/pvrcnn_source/default/ckpt/checkpoint_epoch_6.pth

CFG_FILE=./cfgs/DA/phase${N}_vehicle_${PLATFORM}/st3d/pvrcnn_st3d.yaml

if [ ${NUM_GPUs} -gt 1 ]; then
    echo "Running with ${NUM_GPUs} GPUs..."
    bash scripts/dist_train_uda.sh ${NUM_GPUs} \
        --cfg_file ${CFG_FILE} \
        --pretrained_model ${PRETRAINED_MODEL}
else
    echo "Running with single GPU..."
    python train_uda.py \
        --cfg_file ${CFG_FILE} \
        --pretrained_model ${PRETRAINED_MODEL}
fi