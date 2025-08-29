#!/bin/bash

NUM_GPUs=${1:-2}

CFG_FILE=./cfgs/mdt3d_models/centerpoint_gblobs_waymo_pi3det.yaml

CKPT=/home/uqhyan14/vlm_challenge/LION/ckpt/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs/checkpoint_epoch_30.pth
CKPT=/home/uqhyan14/vlm_challenge/LION/output/cfgs/mdt3d_models/centerpoint_gblobs_waymo_pi3det/default/ckpt/checkpoint_epoch_30.pth
CKPT=/home/uqhyan14/vlm_challenge/LION/ckpt/mdt3d_models/centerpoint_gblobs_waymo/checkpoint_epoch_30.pth


CFG_FILE=./cfgs/kitti-waymo_models/second_gblobs_3d-vfield_pi3det.yaml
CKPT=/home/uqhyan14/vlm_challenge/LION/output/cfgs/kitti-waymo_models/second_gblobs_3d-vfield_pi3det/default/ckpt/checkpoint_epoch_80.pth



CFG_FILE=./cfgs/kitti-waymo_models/pointpillar_gblobs_3d-vfield_pi3det.yaml
CKPT=/home/uqhyan14/vlm_challenge/LION/output/cfgs/kitti-waymo_models/pointpillar_gblobs_3d-vfield_pi3det/default/ckpt/checkpoint_epoch_80.pth


# CFG_FILE=./cfgs/kitti-waymo_models/PartA2_gblobs_3d-vfiled_pi3det.yaml
# CKPT=/home/uqhyan14/vlm_challenge/LION/output/cfgs/kitti-waymo_models/PartA2_gblobs_3d-vfiled_pi3det/default/ckpt/checkpoint_epoch_80.pth



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