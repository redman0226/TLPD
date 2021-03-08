import os
from os.path import join,basename
import json

import math
import argparse

import torch
from torch.multiprocessing import Queue, Process
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import mmcv

from cwd_utils import *


def main(args):
    if not args.just_calculate:

        torch.set_default_tensor_type('torch.FloatTensor')

        net = network.Network(args)
        net.cuda('cuda')
        check_point = torch.load(args.model_path)
        net.load_state_dict(check_point['state_dict'])

        os.makedirs(args.output_dir, exist_ok=True)

        retDets = []
        for tmp_id,img_file_name in enumerate(os.listdir(args.image_dir)):

            img_path = join(args.image_dir, img_file_name)
            output_image_dir = None
            if args.save_image:
                output_image_dir = join(args.output_dir, 'predicts')
                os.makedirs(output_image_dir, exist_ok=True)

            np.set_printoptions(precision=2, suppress=True)
            try:
                detections = inference_image_path( args.image_dir, img_file_name,net, category_id =  args.category_id)['dtboxes']
            except:
                print('[!] load error ',join(args.image_dir,img_file_name))
                continue


            if args.save_image:
                bboxes, labels = [], []
            for res in detections:
                
                score = res['score']
                if score < args.confidence:
                    continue
                bbox = res['bbox']
                det = {
                    'image_id': tmp_id,
                    'bbox': bbox,
                    'category_id': args.category_id,
                    'score': score,
                }
                retDets.append(det)
                
                if args.save_image:
                    labels.append(args.category_id)
                    bboxes.append([bbox[0], bbox[1],
                                bbox[2]-bbox[0], bbox[3]-bbox[1], score])

            if args.save_image:
                dst_img_path = join(output_image_dir, basename(img_path))
                if not bboxes:
                    bboxes = [[0., 0., 0., 0., 0.]]
                    labels = [args.category_id]
            
                bboxes = np.asarray([xywh2xyxy(det['bbox']) + [det['score']] for det in detections])
                labels = np.asarray([det['category_id'] for det in detections])
                # formatting
                mmcv.imshow_det_bboxes(
                    img_path,
                    bboxes,
                    labels,
                    score_thr=args.confidence,
                    show=False,
                    out_file=dst_img_path
                )
        
        output_json_path = join(args.output_dir, 'detres.json')
        with open(output_json_path, 'w') as json_fp:
            json_str = json.dumps(retDets)
            json_fp.write(json_str)
        
    #fpath = os.path.join(args.output_dir, 'dump_with_gt.odgt')
    #misc_utils.save_json_lines(all_results, fpath)

  
    print('Done')


'''


'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', default=None, type=str)
    #parser.add_argument('--groundtruth_path', '-gt', default=None, type=str)
    parser.add_argument('--image_dir', '-i', default='/home/kv_zhao/datasets/CrowdHuman/Images', type=str)
    parser.add_argument('--output_dir', '-od', default=None, type=str)

    parser.add_argument('-conf', '--confidence', type=float, default=0.0, help='')

    parser.add_argument('--save_image', action='store_true', help='If set, the predicted images will not be saved.')
    parser.add_argument('--just_calculate', action='store_true', help='If set, it will load the previous json file.')
    parser.add_argument('--calculate', action='store_true', help='')
    parser.add_argument('--device', '-d', default='0', type=str)
    parser.add_argument('-cid', '--category_id', type=int, default=1,help='vbox: 1, fbox: 2, hbox: 3')
    add_arguments(parser)

    args = parser.parse_args()
    main(args)

