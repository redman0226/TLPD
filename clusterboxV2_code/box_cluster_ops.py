from pycocotools.coco import COCO
from os.path import join
import os
import numpy as np
import cv2
from tqdm import tqdm
import math
import json
def get_id(name,box_source):
    for k,v in box_source.imgs.items():
        if name in v['file_name']:
            return k
            
def imcolor_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    return labeled_img

def box_basic_constraint(box, imgH,imgW):
    box[0] = box[0] if box[0] > 0 else 0
    box[2] = box[2] if box[2] < imgW else imgW - 1 
    box[1] = box[1] if box[1] > 0 else 0
    box[3] = box[3] if box[3] < imgH else imgH - 1 
    return box

def box_constraint(box, imgH,imgW , inverse=False):
    h = box[3] - box[1]
    w = box[2] - box[0]
    if inverse and h > w:   
        m = box[0] + w //2 
        box[0] = (m - h//2) if (m - h//2) > 0 else 0
        box[2] = (m + h//2) if (m + h//2) < imgW else imgW - 1 
    elif not inverse and h < w:
        m = box[1] + h //2 
        box[1] = (m - w//2) if (m - w//2) > 0 else 0
        box[3] = (m + w//2) if (m + w//2) < imgH else imgH - 1 
    else:
        return box_basic_constraint(box,imgH,imgW)
    return box

def overlap_combination(boxes, thresh = 0.6,imgH = 200,box_groups=None):
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    dense = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    result = []
    result_box_groups = []
    done = []
    #print(order,len(box_groups))
    for box_i in order:
        if  box_i not in done:
            match = []
            for box_j in order:
                if box_i != box_j and box_j not in done:
                    xx1 = np.maximum(x1[box_i], x1[box_j])
                    yy1 = np.maximum(y1[box_i], y1[box_j])
                    xx2 = np.minimum(x2[box_i], x2[box_j])
                    yy2 = np.minimum(y2[box_i], y2[box_j])

                    w = np.maximum(0.0, xx2 - xx1 + 1)
                    h = np.maximum(0.0, yy2 - yy1 + 1)
                    inter = w * h
                    #rate = yy1 / imgH
                    
                    rate = 1
                    if inter / areas[box_i] > (thresh*rate) or inter / areas[box_j] > (thresh*rate):
                        match.append(box_j)
                        done.append(box_j)
            done.append(box_i)
            xmin,ymin,xmax,ymax,dense_ = boxes[box_i]
            if box_groups is not None:
                group = box_groups[box_i]
            for box_match in match:
                xmin = np.minimum(xmin, x1[box_match])
                ymin = np.minimum(ymin, y1[box_match])
                xmax = np.maximum(xmax, x2[box_match])
                ymax = np.maximum(ymax, y2[box_match])
                dense_ += dense[box_match]
                if box_groups is not None:
                    group += box_groups[box_match]
            result.append([xmin,ymin,xmax,ymax,dense_])
            if box_groups is not None:
                result_box_groups.append(group)
    return result if box_groups is None else result,result_box_groups
def dist_(A,B):
    return math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
def combine_box(A,B):
    xmin,ymin,xmax,ymax,dense = A
    xmin = np.minimum(xmin, B[0])
    ymin = np.minimum(ymin, B[1])
    xmax = np.maximum(xmax, B[2])
    ymax = np.maximum(ymax, B[3])
    dense = dense + B[4]
    return [xmin,ymin,xmax,ymax,dense]
def H_W_rate(box):
    return (box[3]-box[1])/(box[2]-box[0])
def dense_constraint_limit_height(boxes,dense_thr = 5, distance_thr = 0, height_weight = 1,width_weight = 1, H_W_rate_upper_lower = (1.5,0.7)  ,skip_thr_check = False, box_groups = None):
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    dense = boxes[:, 4]
    result = []
    result_box_groups = []
    done = []
    order = dense.argsort()[::-1]
    for box_i in order:
        NEW_ONE = True
        if  box_i not in done:
            if box_groups is not None:
                group = box_groups[box_i]
            xmin,ymin,xmax,ymax,dense_ = boxes[box_i]
            if dense[box_i] <= dense_thr:
                record_boxes = []
                distances = [] 
                m1 = [(x1[box_i]+x2[box_i])/2,(y1[box_i]+y2[box_i])/2] 
                for box_j in order:
                    if box_i != box_j and box_j not in done:
                        if dense[box_j] <= dense_thr or skip_thr_check:
                            m2 = [(x1[box_j]+x2[box_j])/2,(y1[box_j]+y2[box_j])/2] 
                            distance = math.sqrt(width_weight * (m1[0]-m2[0])**2 + height_weight * (m1[1]-m2[1])**2)
                            if(distance < distance_thr):
                                rate = H_W_rate(combine_box(boxes[box_i],boxes[box_j])) 
                                if rate < H_W_rate_upper_lower[0] and rate > H_W_rate_upper_lower[1]:
                                    distances.append(distance)
                                    record_boxes.append(box_j)
                distances = np.array(distances)
                if len(distances) > 0:
                    min_distance = record_boxes[np.argmin(distances)]
                    match = [min_distance]
                    for box_match in match:
                        xmin = np.minimum(xmin, x1[box_match])
                        ymin = np.minimum(ymin, y1[box_match])
                        xmax = np.maximum(xmax, x2[box_match])
                        ymax = np.maximum(ymax, y2[box_match])
                        dense_ += dense[box_match]
                        done.append(box_match)
                        if box_groups is not None:
                            group += box_groups[box_match]
                else:
                    for i,r in enumerate(result):
                        if r[4] <= dense_thr or skip_thr_check:
                            m2 = [(r[0]+r[2])/2,(r[1]+r[3])/2] 
                            distance = math.sqrt((m1[0]-m2[0])**2 + (m1[1]-m2[1])**2)
                            if(distance < distance_thr):
                                xmin_ = np.minimum(xmin, r[0])
                                ymin_ = np.minimum(ymin, r[1])
                                xmax_ = np.maximum(xmax, r[2])
                                ymax_ = np.maximum(ymax, r[3])
                                dense_ += r[4]
                                NEW_ONE = False
                                result[i] = [xmin_,ymin_,xmax_,ymax_,dense_]
                                if box_groups is not None:
                                    result_box_groups[i] += group
                                break
            done.append(box_i)
            if NEW_ONE:
                result.append([xmin,ymin,xmax,ymax,dense_])
                if box_groups is not None:
                    result_box_groups.append(group)
    return result if box_groups is None else result,result_box_groups
def extend_constraint(box,imgH,imgW,dense_thr = 10,full_rate = 0.5, inverse = False, height_weight = 1):
    
    if box[4] < dense_thr:
        if inverse:
            rate = full_rate * (1 - box[4]/dense_thr)
        else:
            rate = full_rate * (box[4]/dense_thr)
        ex_h = int((box[3] - box[1])*rate) * height_weight
        m = box[1] + (box[3] - box[1]) //2 
        box[1] -= ex_h
        box[3] += ex_h
        box[1] = box[1] if box[1] > 0 else 0
        box[3] = box[3] if box[3] < imgH else imgH - 1
        
        ex_w = int((box[2] - box[0])*rate)
        m = box[0] + (box[2] - box[0]) //2
        box[0] = box[0] if box[0] > 0 else 0
        box[2] = box[2] if box[2] < imgW else imgW - 1 
    return box

def distnace(A,B):
    return math.sqrt((A['center'][0] - B['center'][0])**2 + (A['center'][1] - B['center'][1])**2)
def remove_low_score(dt_boxes,conf=0.4):
    processing_boxes = []
    for box in dt_boxes:
        if(box['score']>conf):
            box = np.array(box['bbox'])
            midX = box[0] + box[2]//2
            midY = box[1] + box[3]//2
            processing_boxes.append({
                'center' : [midX,midY],
                'bbox' : [box[0],box[1],box[2],box[3]]
            })
    return processing_boxes
def cluster_by_width(args, processing_boxes, factor = 1):
    box_groups = []
    done = []
    for i,i_pbox in enumerate(processing_boxes):
        if i not in done:
            current_group = [i_pbox]
            done.append(i)
            for j,j_pbox in enumerate(processing_boxes):
                if j not in done:
                    for c,c_box in enumerate(current_group):
                        #print(c_box)
                        threshold = (c_box['bbox'][2] + j_pbox['bbox'][2]) * factor
                        if distnace(c_box, j_pbox) < threshold:
                            current_group += [j_pbox]
                            done += [j]
                            break
            box_groups += [current_group]
    
    bbox_4m = []
    
    for group in box_groups:
        minX,minY,w,h = group[0]['bbox']
        maxX = minX + w
        maxY = minY + h
        for k in range(len(group)):
            b = group[k]['bbox']
            if minX > b[0]:
                minX = b[0]
            if minY > b[1]:
                minY = b[1]
            if maxX < b[0]+b[2]:
                maxX = b[0]+b[2]
            if maxY < b[1]+b[3]:
                maxY = b[1]+b[3]
        bbox_4m.append([minX,minY,maxX,maxY,len(group)])
    
    return bbox_4m,box_groups



def _tranfer_back(boxes):
    result = []
    for box in boxes:
        #print(box)
        midX = box[0] + box[2]//2
        midY = box[1] + box[3]//2
        result += [
            {
                'center' : [midX,midY],
                'bbox' : [box[0],box[1],box[2],box[3]]
            }
        ]
    return result
def process_cluster(args,processing_boxes,height,width,\
    width_cluster_factor = 1, 
    overlap_threshold = 0.5, 
    distance_thr_rate = 0.1, 
    dense_thr=10,
    width_weight = 0.8,
    H_W_rate_upper_lower = (1.5,0.7)):
    
    bbox_4m,box_groups = cluster_by_width(args,processing_boxes,factor=width_cluster_factor)
    
    processing_boxes,box_groups = \
        overlap_combination(bbox_4m,thresh=overlap_threshold,imgH=height,box_groups=box_groups) 
    
    processing_boxes,box_groups = dense_constraint_limit_height(processing_boxes, 
        dense_thr=dense_thr,\
            distance_thr = math.sqrt(height**2+width**2) * distance_thr_rate,\
            width_weight= width_weight,\
            H_W_rate_upper_lower = H_W_rate_upper_lower,\
            box_groups=box_groups)

    processing_boxes,box_groups = \
        overlap_combination(processing_boxes,thresh=overlap_threshold,imgH=height,box_groups=box_groups)

    for box_i in range(len(processing_boxes)):
            processing_boxes[box_i] = box_constraint(processing_boxes[box_i], height, width, inverse=True)
  
    #processing_boxes,box_groups = overlap_combination(processing_boxes,thresh=overlap_threshold,imgH=height,box_groups=box_groups)
 
    for box_i in range(len(processing_boxes)):
        processing_boxes[box_i] = box_constraint(processing_boxes[box_i], height, width, inverse=False)

    #processing_boxes,box_groups = overlap_combination(processing_boxes,thresh=overlap_threshold,imgH=height,box_groups=box_groups)

    return processing_boxes,box_groups

def two_stage_process_cluster(args,dt_boxes,height,width, 
                                width_cluster_factors = [1,1],
                                overlap_thresholds = [0.5,0.5],
                                H_W_rate_upper_lower = [1.5,0.7],
                                distance_thr_rates = [0.1,0.05],
                                sec_stage_threshold = 30
                                ):
    
    processing_boxes = remove_low_score(dt_boxes,conf=args.confidence)
    if len(processing_boxes) < 1:
        #print('[!] Total box:',len(processing_boxes))
        return processing_boxes
    processing_boxes,box_groups = process_cluster(
        args,processing_boxes,height,width,
        width_cluster_factor = width_cluster_factors[0],
        overlap_threshold = overlap_thresholds[0],
        H_W_rate_upper_lower = H_W_rate_upper_lower,
        distance_thr_rate=distance_thr_rates[0])
    iter_size = len(processing_boxes)
    to_remove = []
    if args.two_stage:
        for i in range(iter_size):
            box,group = processing_boxes[i],box_groups[i]
            if box[4] > sec_stage_threshold:
                sub_processing_boxes, sub_box_groups = process_cluster(
                    args,group,height,width,
                    width_cluster_factor = width_cluster_factors[1],
                    overlap_threshold = overlap_thresholds[1],
                    distance_thr_rate = distance_thr_rates[1],
                    H_W_rate_upper_lower = H_W_rate_upper_lower)
                processing_boxes += sub_processing_boxes
                box_groups += sub_box_groups
                to_remove +=[i]
                #for sub_box,sub_group in zip(sub_processing_boxes,sub_box_groups):
                    #if sub_box[4] > sec_stage_threshold:
                    #    r= process_cluster(args,sub_group,height,width,distance_thr_rate = 0.025)[0]
                    #    print(r)
                    #    exit()
                    #    processing_boxes += r
                    #else:
                    #    processing_boxes += sub_box
    result = []
    result_box_groups = []
    for i in range(len(processing_boxes)):
        if  i not in to_remove:
            result.append(processing_boxes[i])
            result_box_groups.append(box_groups[i])
    
    return result,result_box_groups