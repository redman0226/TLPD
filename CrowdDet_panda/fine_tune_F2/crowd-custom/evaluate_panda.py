import os
from os.path import join,basename
import json

import math
import argparse

import torch
from torch.multiprocessing import Queue, Process
import numpy as np
from tqdm import tqdm

from cwd_utils import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import mmcv

def load_json(file):
    with open(file, 'r') as fp:
        jsonfile = json.load(fp)
    return jsonfile


def main(args):
    print('[!]Loading model :D')
    torch.set_default_tensor_type('torch.FloatTensor')
    net = network.Network(args)
    net.cuda('cuda')
    check_point = torch.load(args.model_path)
    net.load_state_dict(check_point['state_dict'])

    os.makedirs(args.output_dir, exist_ok=True)
    retDets = []
    tq = tqdm(enumerate(os.listdir(args.image_dir)))
    split_infos = load_json(args.split_json)

    
    print('[!]Loading gt :P')
    cocoGt = COCO(args.gt_json)
    tq_image_id = tqdm(cocoGt.getImgIds())

    print('[!]Start inferencing!')
    
    def update_desc(str0,str1):
        desc = '[-]{}[{}]'
        tq_image_id.set_description(desc.format(str0,str1))
        tq_image_id.refresh()
    
    retDets = []
    for image_id in tq_image_id:
        img_file_name = cocoGt.loadImgs(ids=[image_id])[0]['file_name']
        
        image = cv2.imread(join(args.image_dir,img_file_name), cv2.IMREAD_COLOR)
        try:
            info_ = split_infos[img_file_name.replace('Station_Square','Station Square')]
        except:
            info_ = split_infos[img_file_name]

        #print(image_id)
        #print(img_file_name)
        for i,split_box in enumerate(info_):
            update_desc(img_file_name,i)
            patch = image[split_box[1]:split_box[3],split_box[0]:split_box[2],:]
            detections = inference_single_extend(patch,net,args.category_id,fname=img_file_name)['dtboxes']
            for res in detections:
                score = res['score']
                if score < args.confidence:
                    continue
                bbox = res['bbox']
                det = {
                    'image_id': image_id,
                    'bbox': [
                        bbox[0]+split_box[0],bbox[1]+split_box[1],
                        bbox[2],bbox[3]],
                    'category_id': args.category_id,
                    'score': score,
                }
                retDets.append(det)
                
    output_json_path = join(args.output_dir, 'detres.json')
    with open(output_json_path, 'w') as json_fp:
        json_str = json.dumps(retDets)
        json_fp.write(json_str)
    print('Done')
       

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', default='', type=str)

    
    parser.add_argument('--gt_json', '-gt', default='', type=str)

    parser.add_argument('--split_json', '-sj', default='', type=str)
    parser.add_argument('--image_dir', '-i', default='', type=str)
    parser.add_argument('--output_dir', '-od', default='', type=str)



    parser.add_argument('-conf', '--confidence', type=float, default=0.05, help='')
    parser.add_argument('--save_image', action='store_true', help='If set, the predicted images will not be saved.')
    parser.add_argument('--device', '-d', default='0', type=str)
    parser.add_argument('-cid', '--category_id', type=int, default=2,help='vbox: 1, fbox: 2, hbox: 3')
    add_arguments(parser)

    args = parser.parse_args()
    main(args)
