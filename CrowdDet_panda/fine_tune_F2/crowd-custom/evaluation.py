import os
from os.path import join,basename
import json

import math
import argparse

import torch
from torch.multiprocessing import Queue, Process
import numpy as np
from tqdm import tqdm

import network
import dataset

from config import config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import mmcv
if_set_nms = True

catmap = {
    1: 'vbox',
    2: 'fbox',
    3: 'hbox',
    4: 'vehicle'
}

def xywh2xyxy(_bbox):
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] + _bbox[0],
        _bbox[3] + _bbox[1],]
def get_image(path,fname):
    image = cv2.imread(join(path,fname), cv2.IMREAD_COLOR)
    if config.eval_resize == False:
        resized_img, scale = image, 1
    else:
        resized_img, scale = dataset.resize_img_by_short_and_max_size(
            image, config.eval_image_short_size, config.eval_image_max_size)
    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    transposed_img = np.ascontiguousarray(
        resized_img.transpose(2, 0, 1)[None, :, :, :],
        dtype=np.float32)
    im_info = np.array([height, width, scale, original_height, original_width],
                       dtype=np.float32)
    image = torch.Tensor(transposed_img).cuda(0)
    im_info = torch.Tensor(im_info[None, :]).cuda(0)
    return image,im_info
def boxes_dump(boxes, is_gt):
    result = []
    boxes = boxes.tolist()
    for box in boxes:
        if is_gt:
            box_dict = {}
            box_dict['bbox'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box_dict['tag'] = box[-1]
            result.append(box_dict)
        else:
            def s(f):
                return float("{:.2f}".format(f))
            box_dict = {}
            box_dict['bbox'] = [s(box[0]), s(box[1]), s(box[2]-box[0]), s(box[3]-box[1])]
            #box_dict['tag'] = 1
            box_dict['proposal_num'] = box[-1]
            box_dict['score'] = float("{:.2f}".format(box[-2]))
            box_dict['category_id'] = 1
            result.append(box_dict)
    return result
def inference_single(net, path,fname, device):
    image,im_info = get_image(path,fname)
    np.set_printoptions(precision=2, suppress=True)
    net.eval()
    pred_boxes = net(image, im_info)
    
    if if_set_nms:
        from set_nms_utils import set_cpu_nms
        n = pred_boxes.shape[0] // 2
        idents = np.tile(np.arange(n)[:,None], (1, 2)).reshape(-1, 1)
        pred_boxes = np.hstack((pred_boxes, idents))
        keep = pred_boxes[:, -2] > 0.05
        pred_boxes = pred_boxes[keep]
        keep = set_cpu_nms(pred_boxes, 0.5)
        pred_boxes = pred_boxes[keep]
    else:
        import det_tools_cuda as dtc
        nms = dtc.nms
        keep = nms(pred_boxes[:, :4], pred_boxes[:, 4], 0.5)
        pred_boxes = pred_boxes[keep]
        pred_boxes = np.array(pred_boxes)
        keep = pred_boxes[:, -1] > 0.05
        pred_boxes = pred_boxes[keep]
    result_dict = dict(fname=fname, height=int(im_info[0, -2]), width=int(im_info[0, -1]),
            dtboxes=boxes_dump(pred_boxes, False))
            #rois=misc_utils.boxes_dump(rois[:, 1:], True))
    return result_dict
def main(args):
    cocoGt = COCO(args.groundtruth_path)
    if not args.just_calculate:
        # coco format gt
        torch.set_default_tensor_type('torch.FloatTensor')

        net = network.Network(args)
        net.cuda('cuda')
        check_point = torch.load(args.model_path)
        net.load_state_dict(check_point['state_dict'])

        os.makedirs(args.output_dir, exist_ok=True)

        retDets = []
        for image_id in tqdm(cocoGt.getImgIds()):
            img_file_name = cocoGt.loadImgs(ids=[image_id])[0]['file_name']
            img_path = join(args.image_dir, img_file_name)
            output_image_dir = None
            if args.save_image:
                output_image_dir = join(args.output_dir, 'predicts')
                os.makedirs(output_image_dir, exist_ok=True)

            np.set_printoptions(precision=2, suppress=True)
            detections = inference_single(net,args.image_dir,img_file_name,args.device)['dtboxes']

            if args.save_image:
                bboxes, labels = [], []
            for res in detections:
                
                score = res['score']
                if score < args.confidence:
                    continue
                bbox = res['bbox']
                det = {
                    'image_id': image_id,
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
        json_dict = {
            'images': cocoGt.loadImgs(cocoGt.getImgIds()),
            'annotations': retDets,
            'categories': [{
                'supercategory': 'person',
                'id': args.category_id,
                'name': catmap[args.category_id],
                }
            ],
        }
        output_json_path = join(args.output_dir, 'predictions.json')
        with open(output_json_path, 'w') as json_fp:
            json_str = json.dumps(json_dict)
            json_fp.write(json_str)
        output_json_path = join(args.output_dir, 'detres.json')
        with open(output_json_path, 'w') as json_fp:
            json_str = json.dumps(retDets)
            json_fp.write(json_str)
        
    #fpath = os.path.join(args.output_dir, 'dump_.json'))
    #misc_utils.save_json_lines(all_results, fpath)

    if args.calculate:
        cocoDt = cocoGt.loadRes(join(args.output_dir, 'detres.json'))
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.maxDets = [100, 500, 1000]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    print('Done')



