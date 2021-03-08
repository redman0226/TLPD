import os
import math
import argparse

import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from os.path import join
import json



def main(args):
    print('[*] Reading lines to dict :', args.detfile)
    with open(args.detfile, "r") as f:
        lines = f.readlines()
        records = []
        for line in tqdm(lines):
            records.append(json.loads(line.strip('\n')))
    cocoGt = COCO(args.base_json)
    img_id_dict = {}

    print('[*] Recording the image ids.')
    for image_id in tqdm(cocoGt.getImgIds()):
        img_id_dict[cocoGt.loadImgs(ids=[image_id])[0]['file_name'].split('.')[0]] = image_id

    print('[*] Start the coverting process...')
    retDets = []
    for record in tqdm(records):
        for box in record['dtboxes']:
            det = {
                        'image_id': img_id_dict[record['ID']],
                        'bbox': box['box'],
                        'category_id': 1,
                        'score': box['score']
                    }
            retDets.append(det)
    
    print('[*] Writing the result:',join(args.output_dir, args.output_name))
    output_json_path = join(args.output_dir, args.output_name)
    with open(output_json_path, 'w') as json_fp:
            json_str = json.dumps(retDets)
            json_fp.write(json_str)
    print('[!] Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser.add_argument("--detfile",'-d',default='../outputs/eval_dump/dump-emd_refine.pth.json',help='The one to be converted.', type=str)
    parser.add_argument("--base_json",'-b',default='/home/kv_zhao/datasets/CrowdHuman/annotations/val.json',help='To get the needed image id map.', type=str)
    parser.add_argument("--output_dir",'-o',default='./', type=str)
    parser.add_argument("--output_name",'-on',default='detres_emd_refine.json', type=str)
    args = parser.parse_args()
    main(args)
    