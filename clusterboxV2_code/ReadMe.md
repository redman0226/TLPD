require packages :
 pycocotools,numpy,tqdm,cv2

examples:
 python box_cluster.py --box_json ./train_predictions.json -od ./res_train
 python box_cluster.py --box_json ./test_predictions.json -od ./res_test
 