from box_cluster_ops import *

def main(args):
    
    if args.box_det is not None:
        print('[!]Read COCO ',args.box_json,'->', args.box_det)
        box_source = COCO(args.box_json).loadRes(args.box_det)

    else:
        print('[!]Read COCO ',args.box_json)
        box_source = COCO(args.box_json)


    
    os.makedirs(args.output_dir,exist_ok=True)
    with open(join(args.output_dir,'args.txt'),'w') as f:
        for arg in args.__dict__:
            f.write(str(arg)+' = '+ str(args.__dict__[arg])+' \n')

    output_dir_upper = args.output_dir
    args.output_dir = join(args.output_dir,'draw_box_output')
    os.makedirs(args.output_dir,exist_ok=True)
    print('[-] Total images :',len(box_source.imgs.items()))
    boxes_dict = {}

    ps = tqdm(box_source.imgs.items())
    Counter = 0
    #print(len(box_source.imgs.items()))
    for image_id,image_info in ps:
        #print(image_info)
        #print(image_id)
        image_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        #if '01' in image_name:
        #    continue
        ps.set_description('[-] Proccessing :'+ image_name)
        ps.refresh()

        dt_boxes = np.array(box_source.loadAnns(box_source.getAnnIds([image_id])))
        boxes,box_groups = two_stage_process_cluster(
            args, dt_boxes, height, width,
            width_cluster_factors =[args.width_cluster_factor1,args.width_cluster_factor2],
            overlap_thresholds = [args.overlap_threshold1, args.overlap_threshold2],
             H_W_rate_upper_lower = [args.HW_upper,args.HW_lower],
             distance_thr_rates=[args.distance_thr_rates_s1,args.distance_thr_rates_s2])
    
        if args.draw_predict:
            image = cv2.imread(join(args.image_dir,image_name))
            os.makedirs(join(args.output_dir,image_name),exist_ok=True)

        for h,box in enumerate(boxes):
            #print(box)
            boxes[h] = extend_constraint(box,height,width,dense_thr = 5,full_rate = args.extend_c,inverse=args.extend_inverse)
            box = boxes[h]
            
            if args.draw_predict:
                image = cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,225,0),15)
                cv2.putText(image, 'dense:'+str(box[4]), (int(box[2]),int(box[3])), cv2.FONT_HERSHEY_SIMPLEX,10, (0,225,0), 10, cv2.LINE_AA)

            boxes[h] = [int(x) for x in box[:4]]
            Counter +=1 

        if args.draw_predict:
            cv2.imwrite('{}/boxed_im.jpg'.format(join(args.output_dir,image_name)),image)   
        if len(boxes) > 0:
            boxes_dict[image_name] = boxes

    with open(join(output_dir_upper,'split_boxbased.json'), 'w') as json_file:
        json.dump(boxes_dict, json_file)
    if args.analyze:
        print('[!] \n\nTotal box:' + str(Counter))
        area_ratio_mean = 0
        c = 0
        for box,group in zip(boxes,box_groups):
            area = (box[2]-box[0])*(box[3]-box[1])
            for sub_box in group:
                area_ratio_mean += (sub_box['bbox'][2]*sub_box['bbox'][3]/area)
                c += 1
        print('[!] Each box area mean:' + str(area_ratio_mean/c))
        split_box_map = np.zeros([width,height])
        mean_split_dense = 0
        greater_count = 0
        for box in boxes:
            split_box_map[box[0]:box[2],box[1]:box[3]] += 1
        for group in box_groups:
            for sub_box in group:
                sub_box = sub_box['bbox']
                max_split_dense = np.max(split_box_map[sub_box[0]:(sub_box[0]+sub_box[2]),sub_box[1]:(sub_box[1]+sub_box[3])]) 
                mean_split_dense += max_split_dense
                if max_split_dense > 1:
                    greater_count+= 1 
        print('[!] Mean_split_dense:' + str(mean_split_dense/c))
        print('[!] Greater_count:' + str(greater_count))
        


            
           
if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()


    args.add_argument('--box_json', type=str,default='./train_predictions.json')
    #args.add_argument('--box_json', type=str,default='./test_predictions.json')
    args.add_argument('--box_det', type=str,default=None)


    args.add_argument('-od', '--output_dir', type=str, default='./res', help='Output directory')

    args.add_argument('--draw_predict',action='store_true', help='save the result image, in case you have the merged images.')
    args.add_argument('--analyze',action='store_true', help='analyze')
    args.add_argument('-id', '--image_dir', type=str, default=None, help='Dir of the merged images.')
    #0
    args.add_argument('-conf', '--confidence', type=float, default=0.1, help='')
    #1
    args.add_argument('-wcf1','--width_cluster_factor1',type=float,default=1)
    args.add_argument('-wcf2','--width_cluster_factor2',type=float,default=1)
    #2
    args.add_argument('-ort1','--overlap_threshold1',type=float,default=0.5)
    args.add_argument('-ort2','--overlap_threshold2',type=float,default=0.5)
    #3
    args.add_argument('-dtr1','--distance_thr_rates_s1',type=float,default=0.1)
    args.add_argument('-dtr2','--distance_thr_rates_s2',type=float,default=0.05)
    args.add_argument('--HW_upper',type=float,default=1.5)
    args.add_argument('--HW_lower',type=float,default=0.7)

    args.add_argument('--extend_c',type=float,default=0.6)
    args.add_argument('--extend_inverse',action='store_true',help='At the final extend stage, the extend strategy will change from less smaller to less larger.')
    args.add_argument('--two_stage',type=bool,default=True)

    args = args.parse_args()
    main(args)