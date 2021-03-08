# encoding: utf-8
import os
from os.path import join
import sys

import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = '../../'
add_path(os.path.join(root_dir))
add_path(os.path.join(root_dir, 'lib'))
add_path(os.path.join(root_dir, 'utils'))

class Crowd_human:
    class_names = ['background', 'person']
    num_classes = len(class_names)
    root = '/home/kv_zhao/datasets'
    
    root_folder = join(root,'CrowdHuman')
    image_folder = join(root_folder,'Images')
    train_source = join(root_folder,'annotation_train.odgt')
    eval_source = join(root_folder,'annotation_val.odgt')

class Config:
    output_dir = join(root_dir,'outputs')
    
    #model_dir = os.path.join(output_dir, 'model_dump')
    
    model_dir = join(root_dir,'pths')
    eval_dir = os.path.join(output_dir, 'eval_dump')
    
    #init_weights = join(root_dir,'pths/emd_refine.pth')
    init_weights = 'init_weight/R-50.pkl'
    program_name = 'crowdhuman'

    # ----------data config---------- #
    image_mean = np.array([102.9801, 115.9465, 122.7717])#bgr
    train_image_short_size = 800
    train_image_max_size = 1400
    eval_resize = True
    eval_image_short_size = 800
    eval_image_max_size = 1400
    seed_dataprovider = 3
    train_source = Crowd_human.train_source
    eval_source = Crowd_human.eval_source
    image_folder = Crowd_human.image_folder
    class_names = Crowd_human.class_names
    num_classes = Crowd_human.num_classes
    class_names2id = dict(list(zip(class_names, list(range(num_classes)))))
    gt_boxes_name = 'fbox'

    # ----------train config---------- #
    bn_training = False
    train_batch_per_gpu = 2
    momentum = 0.9
    weight_decay = 1e-4
    base_lr = 1e-3 * train_batch_per_gpu * 1.25
    warm_iter = 800
    lr_decay = [300000, 390000]
    train_base_iters = 450000
    model_dump_interval = 15000
    log_dump_interval = 20

    # ----------test config---------- #
    test_cls_threshold = 0.05
    test_nms_version = 'original'
    test_max_boxes_per_image = 300 #200
    test_save_type = 'human'
    test_nms = 0.5
    test_vis_threshold = 0.3

    # ----------model config---------- #
    batch_filter_box_size = 0
    nr_box_dim = 5
    ignore_label = -1
    max_boxes_of_image = 500

    # ----------rois generator config---------- #
    anchor_base_size = 16
    anchor_scales = np.array([2, 4, 8, 16, 32])
    anchor_aspect_ratios = [1, 2, 3]
    num_anchor_scales = len(anchor_scales)
    num_cell_anchors = len(anchor_aspect_ratios)
    anchor_within_border = False

    rpn_min_box_size = 2
    rpn_nms_threshold = 0.7
    train_prev_nms_top_n = 12000
    train_post_nms_top_n = 2000
    test_prev_nms_top_n = 6000
    test_post_nms_top_n = 1000

    # ----------binding&training config---------- #
    rpn_smooth_l1_beta = 1
    rcnn_smooth_l1_beta = 1

    num_sample_anchors = 256
    positive_anchor_ratio = 0.5
    rpn_positive_overlap = 0.7
    rpn_negative_overlap = 0.3
    rpn_bbox_normalize_targets = False

    num_rois = 512
    fg_ratio = 0.5
    fg_threshold = 0.5
    bg_threshold_high = 0.5
    bg_threshold_low = 0.0
    rcnn_bbox_normalize_targets = True
    bbox_normalize_means = np.array([0, 0, 0, 0])
    bbox_normalize_stds = np.array([0.1, 0.1, 0.2, 0.2])

config = Config()

