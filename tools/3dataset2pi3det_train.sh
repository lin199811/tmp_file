#!/bin/bash

NUM_GPUs=${1:-2}

CFG_FILE=./cfgs/mdt3d_models/centerpoint_gblobs_waymo_pi3det.yaml

PRETRAINED_MODEL=/home/uqhyan14/vlm_challenge/LION/ckpt/mdt3d_models/centerpoint_gblobs_waymo/checkpoint_epoch_30.pth


CFG_FILE=./cfgs/kitti-waymo_models/second_gblobs_3d-vfield_pi3det.yaml
PRETRAINED_MODEL=/home/uqhyan14/vlm_challenge/LION/ckpt/kitti-waymo_models/second_gblobs_3d-vfield/checkpoint_epoch_80.pth


CFG_FILE=./cfgs/kitti-waymo_models/pointpillar_gblobs_3d-vfield_pi3det.yaml
PRETRAINED_MODEL=/home/uqhyan14/vlm_challenge/LION/ckpt/kitti-waymo_models/pointpillar_gblobs_3d-vfield/checkpoint_epoch_80.pth


# CFG_FILE=./cfgs/kitti-waymo_models/PartA2_gblobs_3d-vfiled_pi3det.yaml
# PRETRAINED_MODEL=/home/uqhyan14/vlm_challenge/LION/ckpt/kitti-waymo_models/PartA2_gblobs_3d-vfiled/checkpoint_epoch_80.pth



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