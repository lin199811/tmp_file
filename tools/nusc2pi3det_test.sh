#!/bin/bash

NUM_GPUs=${1:-2}

PLATFORM=quadruped
N=2

CFG_FILE=./cfgs/DA/phase${N}_vehicle_${PLATFORM}/source_only/pvrcnn_st3d.yaml

CFG_FILE=./cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det_l40.yaml
CFG_FILE=./cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det.yaml

# CKPT=/home/uqhyan14/vlm_challenge/LION/ckpt/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs/checkpoint_epoch_30.pth
# CKPT=/home/uqhyan14/vlm_challenge/LION/output/cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det/default/ckpt/checkpoint_epoch_30.pth
# CKPT=/home/uqhyan14/vlm_challenge/LION/output/cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det_l40/default/ckpt/checkpoint_epoch_30.pth


CFG_FILE=./cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det_st3d.yaml
CKPT=/home/uqhyan14/vlm_challenge/LION/output/cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det_st3d/default/ckpt/checkpoint_epoch_4.pth
# CKPT=/home/uqhyan14/vlm_challenge/LION/output/cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det_l40/default/ckpt/checkpoint_epoch_30.pth

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