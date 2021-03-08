
import network
import dataset
import torch
from config import config
import numpy as np
import cv2
from set_nms_utils import set_cpu_nms,cpu_nms
from os.path import join
if_set_nms = True


def xywh2xyxy(_bbox):
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] + _bbox[0],
        _bbox[3] + _bbox[1],]
def add_arguments(parser):
    parser.add_argument('--flip_JSD', action='store_true',help='Semi-supervised learning enable.')
    parser.add_argument('--flip_JSD_0g', action='store_true',help='Semi-supervised learning enable with zero grad on flipped input.')
    parser.add_argument('--recursive', action='store_true',help='recursive mode enable.')
    parser.add_argument('--diff_loss', action='store_true',help='Diff_loss learning enable.')
    
    parser.add_argument('--ensembleRCNN', action='store_true',help='EnsembleRCNN learning enable.')
    parser.add_argument('--bit', action='store_true',help='group_norm ResNet50 enable.')
    parser.add_argument('--resnext50_32x4d', action='store_true',help='ResNeXt50 enable.')
    parser.add_argument('--resnext101_32x8d', action='store_true',help='ResNeXt101 enable.')
    parser.add_argument('--flip_aug', action='store_true',help='')
    parser.add_argument('--FA_heavy', action='store_true',help='heavy flip_aug')
    parser.add_argument('--multiscale','-ms', action='store_true',help='inference with multiscale')

def boxes_dump(boxes, is_gt, category_id = 1):
    result = []
    boxes = boxes.tolist()
    for box in boxes:
        if is_gt:
            box_dict = {}
            box_dict['bbox'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box_dict['tag'] = box[-1]
            result.append(box_dict)
        else:
            box_dict = {}
            box_dict['bbox'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            #box_dict['tag'] = 1
            box_dict['proposal_num'] = box[-1]
            box_dict['score'] = box[-2]
            box_dict['category_id'] = category_id
            result.append(box_dict)
    return result

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

def get_image_info(image, max_size = -1, short_size = -1):
    if config.eval_resize == False:
        resized_img, scale = image, 1
    else:
        max_size = config.eval_image_max_size if max_size < 1 else max_size
        short_size = config.eval_image_short_size if short_size < 1 else short_size
        resized_img, scale = dataset.resize_img_by_short_and_max_size(
            image, short_size, max_size)
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

def __inference(image,im_info,net):
    pred_boxes = net(image, im_info)
    n = pred_boxes.shape[0] // 2
    idents = np.tile(np.arange(n)[:,None], (1, 2)).reshape(-1, 1)
    
    pred_boxes = np.hstack((pred_boxes, idents))
    keep = pred_boxes[:, -2] > 0.05
    pred_boxes = pred_boxes[keep]
    keep = set_cpu_nms(pred_boxes, 0.5)
    pred_boxes = pred_boxes[keep]
    return pred_boxes
def inference_single_extend(in_image, net, category_id = 0, flip_test = False, Multiscale = [], simplified = False, fname = None):
    #np.set_printoptions(precision=2, suppress=True)
    net.eval()
    image, im_info = get_image_info(in_image)
    pred_boxes = __inference(image,im_info,net)

    if len(Multiscale) > 0:
        for scaling in Multiscale:
            max_size = int(config.eval_image_max_size * scaling)
            short_size = int(config.eval_image_short_size * scaling)
            image,im_info = get_image_info(in_image,max_size=max_size,short_size=short_size)
            pred_boxes = np.concatenate([pred_boxes,__inference(image,im_info,net)],axis=0)
        keep = cpu_nms(pred_boxes,0.5)
        pred_boxes = pred_boxes[keep]

    if flip_test:
        pred_boxes_f = __inference(image.flip(-1),im_info,net)
        for i,box_f in enumerate(pred_boxes_f):
            pred_boxes_f[i] = [im_info[0, -1] - box_f[2],box_f[1],im_info[0, -1] - box_f[0],box_f[3],box_f[4],box_f[5]]
        pred_boxes = np.concatenate([pred_boxes,pred_boxes_f],axis=0)
        keep = cpu_nms(pred_boxes,0.5)
        pred_boxes = pred_boxes[keep]

    if simplified:
        return boxes_dump(pred_boxes,False,category_id)
    else:
        return dict(fname=fname, height=int(im_info[0, -2]), width=int(im_info[0, -1]),
            dtboxes=boxes_dump(pred_boxes, False))

def inference_image_path(imdir, fname, net, category_id = 0, flip_test = False, Multiscale = [], simplified = False):
    #np.set_printoptions(precision=2, suppress=True)

    return inference_single_extend(
        cv2.imread(join(imdir,fname)),
        net,
        category_id = category_id, flip_test = flip_test, Multiscale = Multiscale, simplified = simplified)

def inference_single(net,fname, img, device):
    image,im_info = get_image_info(img)
    np.set_printoptions(precision=2, suppress=True)
    net.eval()
    pred_boxes = net(image, im_info)
    
    n = pred_boxes.shape[0] // 2
    idents = np.tile(np.arange(n)[:,None], (1, 2)).reshape(-1, 1)
    pred_boxes = np.hstack((pred_boxes, idents))
    keep = pred_boxes[:, -2] > 0.05
    pred_boxes = pred_boxes[keep]
    keep = set_cpu_nms(pred_boxes, 0.5)
    pred_boxes = pred_boxes[keep]
    
    result_dict = dict(fname=fname, height=int(im_info[0, -2]), width=int(im_info[0, -1]),
            dtboxes=boxes_dump(pred_boxes, False))
    return result_dict

def inference_single_simple(image, net, category_id = 0):
    image,im_info = get_image_info(image)
    np.set_printoptions(precision=2, suppress=True)
    net.eval()

    pred_boxes = net(image, im_info)
    
    n = pred_boxes.shape[0] // 2
    idents = np.tile(np.arange(n)[:,None], (1, 2)).reshape(-1, 1)
    pred_boxes = np.hstack((pred_boxes, idents))
    keep = pred_boxes[:, -2] > 0.05
    pred_boxes = pred_boxes[keep]
    keep = set_cpu_nms(pred_boxes, 0.5)
    pred_boxes = pred_boxes[keep]

    return boxes_dump(pred_boxes,False,category_id)


def process_training_data(args,data_generater, num_gpus, to_cuda = True):
    images = []
    gt_boxes = []
    im_info = []
    masks = []
    aug_images = []
    done_an_epoch = False
    for _ in range(num_gpus):
        data = data_generater.__next__()
        minibt_img, minibt_gt, minibt_info = \
                data['data'], data['boxes'], data['im_info']
        images.append(minibt_img)
        gt_boxes.append(minibt_gt)
        im_info.append(minibt_info)
        if args.var_loss:
            masks.append(data['masks'])
        if args.flip_aug:
            aug_images.append(data['aug_data'])
        if data['done']:
            done_an_epoch = True
    if not to_cuda:
        images = torch.Tensor(np.vstack(images))
        gt_boxes = torch.Tensor(np.vstack(gt_boxes))
        im_info = torch.Tensor(np.vstack(im_info))
        extra = {}
        if args.var_loss:
            extra['masks'] = torch.Tensor(np.vstack(masks))
        if args.flip_aug:
            extra['aug_data'] = torch.Tensor(np.vstack(aug_images))
        return images, gt_boxes, im_info, done_an_epoch, extra

    images = torch.Tensor(np.vstack(images)).cuda()
    gt_boxes = torch.Tensor(np.vstack(gt_boxes)).cuda()
    im_info = torch.Tensor(np.vstack(im_info)).cuda()
    extra = {}
    if args.var_loss:
        extra['masks'] = torch.Tensor(np.vstack(masks)).cuda()
    if args.flip_aug:
        extra['aug_data'] = torch.Tensor(np.vstack(aug_images)).cuda()
    return images, gt_boxes, im_info, done_an_epoch, extra