import torch
import os
import glob
import tqdm
import yaml
import numpy as np
import torch.distributed as dist
from pcdet.config import cfg
from pcdet.utils.common_utils import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils, memory_ensemble_utils
import pickle as pkl
import re
from pcdet.models.model_utils.dsnorm import set_ds_target
# from pcdet.utils import tracker_utils, ms3d_utils
from collections import defaultdict
#PSEUDO_LABELS = {}
from multiprocessing import Manager

PSEUDO_LABELS = Manager().dict() #for multiple GPU training
NEW_PSEUDO_LABELS = {}


def check_already_exsit_pseudo_label(ps_label_dir, start_epoch):
    """
    if we continue training, use this to directly
    load pseudo labels from exsiting result pkl

    if exsit, load latest result pkl to PSEUDO LABEL
    otherwise, return false and

    Args:
        ps_label_dir: dir to save pseudo label results pkls.
        start_epoch: start epoc
    Returns:

    """
    # support init ps_label given by cfg
    if start_epoch == 0 and cfg.SELF_TRAIN.get('INIT_PS', None):
        if os.path.exists(cfg.SELF_TRAIN.INIT_PS):
            print ("********LOADING PS FROM:", cfg.SELF_TRAIN.INIT_PS)
            init_ps_label = pkl.load(open(cfg.SELF_TRAIN.INIT_PS, 'rb'))
            PSEUDO_LABELS.update(init_ps_label)

            if cfg.LOCAL_RANK == 0:
                ps_path = os.path.join(ps_label_dir, "ps_label_e0.pkl")
                with open(ps_path, 'wb') as f:
                    pkl.dump(PSEUDO_LABELS, f)

            return cfg.SELF_TRAIN.INIT_PS

    ps_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_label_e*.pkl'))
    if len(ps_label_list) == 0:
        return

    ps_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in ps_label_list:
        num_epoch = re.findall('ps_label_e(.*).pkl', cur_pkl)
        assert len(num_epoch) == 1

        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_ps_label = pkl.load(open(cur_pkl, 'rb'))
            PSEUDO_LABELS.update(latest_ps_label)
            analyze_pseudo_labels(PSEUDO_LABELS)
            return cur_pkl

    return None


def save_pseudo_label_epoch(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch):
    """
    Generate pseudo label with given model.

    Args:
        model: model to predict result for pseudo label
        val_loader: data_loader to predict pseudo label
        rank: process rank
        leave_pbar: tqdm bar controller
        ps_label_dir: dir to save pseudo label
        cur_epoch
    """
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='generate_ps_e%d' % cur_epoch, dynamic_ncols=True)

    pos_ps_meter = common_utils.AverageMeter()
    ign_ps_meter = common_utils.AverageMeter()

    # Since the model is eval status, some object-level data augmentation methods such as 
    # 'random_object_rotation', 'random_object_scaling', 'normalize_object_size' are not used 
    if cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    model.eval()

    for cur_it in range(total_it_each_epoch):

        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)
        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            pred_dicts, ret_dict = model(target_batch)
        if cur_it==1:
            for pred in pred_dicts:
                print("pred_scores: ",pred["pred_scores"])
                print("pred_labels: ",pred["pred_labels"])
        pos_ps_batch, ign_ps_batch = save_pseudo_label_batch(
            target_batch, pred_dicts=pred_dicts,
            need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', None) and
                        cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                        cur_epoch > 0)
        )

        # log to console and tensorboard
        pos_ps_meter.update(pos_ps_batch)
        ign_ps_meter.update(ign_ps_batch)
        disp_dict = {'pos_ps_box': "{:.3f}({:.3f})".format(pos_ps_meter.val, pos_ps_meter.avg),
                     'ign_ps_box': "{:.3f}({:.3f})".format(ign_ps_meter.val, ign_ps_meter.avg)}

        if rank == 0:
            pbar.update()
            pbar.set_postfix(disp_dict)
            pbar.refresh()

    if rank == 0:
        pbar.close()

    gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch)
    print(len(PSEUDO_LABELS))

def optim_pseudo_label_w_traj(val_loader, rank, ps_label_dir, cur_epoch):

    traj_configs = yaml.load(open('cfgs/DA/nusc_m3ed/pseudo_refine/ps_config.yaml','r'), Loader=yaml.Loader)
    # static_veh
    ps_dict = {}
    for info in val_loader.dataset.infos:
        frame_id = info['frame_id']
        try:
            ps_dict[frame_id] = PSEUDO_LABELS[frame_id]
        except:
            gt_box = np.empty([0,9])
            gt_infos = {
                'gt_boxes': gt_box,
                'cls_scores': np.empty([1]),
                'iou_scores': np.empty([1]),
                'memory_counter': np.zeros(gt_box.shape[0])
            }
            ps_dict[frame_id] = gt_infos
            
    static_veh_trk_cfg = tracker_utils.prepare_track_cfg(traj_configs['TRACKING']['VEH_STATIC'])
    all_veh_trk_cfg = tracker_utils.prepare_track_cfg(traj_configs['TRACKING']['VEH_ALL'])
    static_veh_tracks_world = tracker_utils.get_tracklets(val_loader.dataset, ps_dict, static_veh_trk_cfg, cls_id=1)
    all_veh_tracks_world = tracker_utils.get_tracklets(val_loader.dataset, ps_dict, all_veh_trk_cfg, cls_id=1)

    tracks_veh_all, tracks_veh_static = ms3d_utils.refine_veh_labels(val_loader.dataset,list(ps_dict.keys()),
                                                                    all_veh_tracks_world, 
                                                                    static_veh_tracks_world, 
                                                                    static_trk_score_th=traj_configs['TRACKING']['VEH_STATIC']['RUNNING']['SCORE_TH'],
                                                                    veh_pos_th=traj_configs['PS_SCORE_TH']['POS_TH'][0],
                                                                    refine_cfg=traj_configs['TEMPORAL_REFINEMENT'],
                                                                    save_dir=None)

    final_ps_dict = ms3d_utils.update_ps(val_loader.dataset, ps_dict, tracks_veh_all, tracks_veh_static, tracks_ped=None, 
              veh_pos_th=traj_configs['PS_SCORE_TH']['POS_TH'][0], 
              veh_nms_th=0.05, ped_nms_th=0.5, 
              frame2box_key_static='frameid_to_propboxes', 
              frame2box_key='frameid_to_box', frame_ids=list(ps_dict.keys()))

    NEW_PSEUDO_LABELS.update(final_ps_dict)
    gather_and_dump_pseudo_label_result_sigle_rank(rank, ps_label_dir, cur_epoch)
    print(len(PSEUDO_LABELS))

def analyze_pseudo_labels(PSEUDO_LABELS):
    label_counts = defaultdict(int)
    label_scores = defaultdict(list)

    # 遍历所有 frame
    for frame_id, data in PSEUDO_LABELS.items():
        gt_boxes = data['gt_boxes']  # (N, 9)
        labels = gt_boxes[:, 7].astype(int)
        scores = gt_boxes[:, 8].astype(float)

        for l, s in zip(labels, scores):
            label_counts[l] += 1
            label_scores[l].append(s)

    # 统计结果
    print("\n======= PSEUDO_LABELS 统计结果 =======")
    per_line = 8  # 每行显示几个
    resolution =0.02
     # 打印累计分布
    for label, scores in label_scores.items():

        scores = np.array(scores)
        min_s, max_s = scores.min(), scores.max()
        hist, bin_edges = np.histogram(scores, bins=np.arange(0, 1+resolution, resolution))

        # 只保留落在 min/max 内的区间
        valid = (bin_edges[:-1] >= min_s-resolution) & (bin_edges[1:] <= max_s + resolution)
        hist = hist[valid]
        bin_edges = bin_edges[:-1][valid]

        # 计算累计分布
        cum_counts = np.cumsum(hist[::-1])[::-1]
        total = label_counts[label]

        print(f"=== Label {label} === 数量: {total}")
        # print(f"得分范围: [{min_s:.3f}, {max_s:.3f}]")
        if label == -1:
            continue
        row = []
        for i, (b, c) in enumerate(zip(bin_edges, cum_counts), 1):
            pct = c / total * 100
            row.append(f"{b:5.2f}: {c:6d} ({pct:5.1f}%)")
            if i % per_line == 0:
                print("  ".join(row))
                row = []
        if row:  # 打印最后一行不足 per_line 的
            print("  ".join(row))
        # for b, c in zip(bin_edges, cum_counts):
        #     pct = c / total * 100
        #     print(f"  {b:.2f} - 1: {c} ({pct:.1f}%)")

def gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch):
    commu_utils.synchronize()

    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(NEW_PSEUDO_LABELS)

        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)

        NEW_PSEUDO_LABELS.update(new_pseudo_label_dict)

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_PSEUDO_LABELS, f)

    commu_utils.synchronize()
    PSEUDO_LABELS.clear()
    PSEUDO_LABELS.update(NEW_PSEUDO_LABELS)
    analyze_pseudo_labels(PSEUDO_LABELS)

    NEW_PSEUDO_LABELS.clear()

def gather_and_dump_pseudo_label_result_sigle_rank(rank, ps_label_dir, cur_epoch):

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_PSEUDO_LABELS, f)

    PSEUDO_LABELS.clear()
    PSEUDO_LABELS.update(NEW_PSEUDO_LABELS)
    analyze_pseudo_labels(PSEUDO_LABELS)
    NEW_PSEUDO_LABELS.clear()


def save_pseudo_label_batch(input_dict,
                            pred_dicts=None,
                            need_update=True):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
        need_update: Bool.
            If set to true, use consistency matching to update pseudo label
    """
    pos_ps_meter = common_utils.AverageMeter()
    ign_ps_meter = common_utils.AverageMeter()

    batch_size = len(pred_dicts)
    for b_idx in range(batch_size):
        pred_cls_scores = pred_iou_scores = None
        if 'pred_boxes' in pred_dicts[b_idx]:
            # Exist predicted boxes passing self-training score threshold
            pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
            pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()
            if 'pred_cls_scores' in pred_dicts[b_idx]:
                pred_cls_scores = pred_dicts[b_idx]['pred_cls_scores'].detach().cpu().numpy()
            if 'pred_iou_scores' in pred_dicts[b_idx]:
                pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()

            # remove boxes under negative threshold
            if cfg.SELF_TRAIN.get('NEG_THRESH', None):
                labels_remove_scores = np.array(cfg.SELF_TRAIN.NEG_THRESH)[pred_labels - 1]
                remain_mask = pred_scores >= labels_remove_scores
                pred_labels = pred_labels[remain_mask]
                pred_scores = pred_scores[remain_mask]
                pred_boxes = pred_boxes[remain_mask]
                if 'pred_cls_scores' in pred_dicts[b_idx]:
                    pred_cls_scores = pred_cls_scores[remain_mask]
                if 'pred_iou_scores' in pred_dicts[b_idx]:
                    pred_iou_scores = pred_iou_scores[remain_mask]

            labels_ignore_scores = np.array(cfg.SELF_TRAIN.SCORE_THRESH)[pred_labels - 1]
            ignore_mask = pred_scores < labels_ignore_scores
            pred_labels[ignore_mask] = -1

            gt_box = np.concatenate((pred_boxes,
                                     pred_labels.reshape(-1, 1),
                                     pred_scores.reshape(-1, 1)), axis=1)

        else:
            # no predicted boxes passes self-training score threshold
            gt_box = np.zeros((0, 9), dtype=np.float32)

        gt_infos = {
            'gt_boxes': gt_box,
            'cls_scores': pred_cls_scores,
            'iou_scores': pred_iou_scores,
            'memory_counter': np.zeros(gt_box.shape[0])
        }

        # record pseudo label to pseudo label dict
        if need_update:
            ensemble_func = getattr(memory_ensemble_utils, cfg.SELF_TRAIN.MEMORY_ENSEMBLE.NAME)
            gt_infos = ensemble_func(PSEUDO_LABELS[input_dict['frame_id'][b_idx]],
                                     gt_infos, cfg.SELF_TRAIN.MEMORY_ENSEMBLE)

        if gt_infos['gt_boxes'].shape[0] > 0:
            ign_ps_meter.update((gt_infos['gt_boxes'][:, 7] < 0).sum())
        else:
            ign_ps_meter.update(0)
        pos_ps_meter.update(gt_infos['gt_boxes'].shape[0] - ign_ps_meter.val)

        NEW_PSEUDO_LABELS[input_dict['frame_id'][b_idx]] = gt_infos

    return pos_ps_meter.avg, ign_ps_meter.avg


def load_ps_label(frame_id):
    """
    :param frame_id: file name of pseudo label
    :return gt_box: loaded gt boxes (N, 9) [x, y, z, w, l, h, ry, label, scores]
    """
    if frame_id in PSEUDO_LABELS:
        gt_box = PSEUDO_LABELS[frame_id]['gt_boxes']
    else:
        # raise ValueError('Cannot find pseudo label for frame: %s' % frame_id)
        gt_box = np.empty([0,9])
        gt_infos = {
            'gt_boxes': gt_box,
            'cls_scores': np.empty([1]),
            'iou_scores': np.empty([1]),
            'memory_counter': np.zeros(gt_box.shape[0])
        }
        PSEUDO_LABELS.update({
            frame_id: gt_infos
        })
    return gt_box
