#!/bin/bash

NUM_GPUs=${1:-2}

CFG_FILE=./cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det.yaml
# CFG_FILE=./cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det_st3d.yaml
# CFG_FILE=./cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det_l40.yaml
CFG_FILE=./cfgs/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs_pi3det_l405f.yaml


PRETRAINED_MODEL=/home/uqhyan14/vlm_challenge/LION/ckpt/nuscenes-kitti_models/voxel_rcnn_with_centerhead_gblobs/checkpoint_epoch_30.pth

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