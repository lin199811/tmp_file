#! /bin/bash
PLATFORM=quadruped
N=2

## nus mamba
# PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES=0 python  test.py \
# --cfg_file ./cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml \
# --ckpt /scratch/project_mnt/S0202/vlm_challenge/LION/tools/ckpt/checkpoint_epoch_36_nus_mamba.pth
## nus mamba
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/DA/phase${N}_vehicle_${PLATFORM}/source_only/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml \
--extra_tag lion_mamba_nusc_8x_1f_1x_one_stride_128dim \
--batch_size 16 --epochs 36 --max_ckpt_save_num 4 --workers 4 --sync_bn
## nus mamba
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
# --nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
# --cfg_file ./cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml \
# --extra_tag lion_mamba_nusc_8x_1f_1x_one_stride_128dim \
# --batch_size 16 --epochs 36 --max_ckpt_save_num 4 --workers 4 --sync_bn


## nus renet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
# --nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
# --cfg_file ./cfgs/lion_models/lion_retnet_nusc_8x_1f_1x_one_stride_128dim.yaml \
# --extra_tag lion_retnet_nusc_8x_1f_1x_one_stride_128dim \
# --batch_size 16 --epochs 36 --max_ckpt_save_num 4 --workers 4 --sync_bn


# ## nus rwkv
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
# --nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
# --cfg_file ./cfgs/lion_models/lion_rwkv_nusc_8x_1f_1x_one_stride_128dim.yaml \
# --extra_tag lion_rwkv_nusc_8x_1f_1x_one_stride_128dim \
# --batch_size 16 --epochs 36 --max_ckpt_save_num 4 --workers 4 --sync_bn


