import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
from config import config
from backbone import resnet50
from backbone import fpn
from det_opr.anchors_generator import generate_anchors_opr
from det_opr.find_top_rpn_proposals import find_top_rpn_proposals
from det_opr.fpn_anchor_target import fpn_anchor_target, fpn_rpn_reshape
from det_opr.fpn_roi_target import fpn_roi_target
from det_opr.loss_opr import smooth_l1_loss
from det_opr.bbox_opr import bbox_transform_inv_opr, bbox_transform_opr
from layers.pooler import ROIPooler
from det_tools.img_utils import pad_tensor_to_multiple_number
from det_tools.load_utils import _init_backbone
import bit_models

import torchvision.models as models
import torchvision.transforms as transforms
#'BiT-M-R50x1'


def default_preprocessing(image):
    return image - torch.tensor(config.image_mean[None, :, None, None],dtype=image.dtype, device=image.device)
def default_torchvision(image):
    image = image[:,[2,1,0],:,:]#bgr -> rgb
    return (image/255 - torch.tensor(np.array([0.485, 0.456, 0.406])[None, :, None, None],dtype=image.dtype, device=image.device))/ torch.tensor(np.array([0.229, 0.224, 0.225])[None, :, None, None],dtype=image.dtype, device=image.device)
def empty_preprocessing(image):
    return image

class ResNeXt(nn.Module):
    def __init__(self,args, pretrained=True):
        super().__init__()
        if args.resnext50_32x4d:
            resnet_module = models.resnext50_32x4d(pretrained=pretrained,progress=True)
        elif args.resnext101_32x8d:
            resnet_module = models.resnext101_32x8d(pretrained=pretrained,progress=True)
        self.body = nn.ModuleList(resnet_module.children())[:-1]
        self.body = nn.Sequential(*self.body)
    def forward(self, x):
        btn_up = []
        for bname, block in self.body.named_children():
            if bname != '8':
                x = block(x)
                if bname in ['4','5','6','7']:
                    btn_up.append(x)
        return btn_up
                

class Network(nn.Module):
    def __init__(self,args,prune_backbone=None):
        super().__init__()
        if prune_backbone is not None:
            self.resnet50 = prune_backbone
            _init_backbone(self.resnet50, config.init_weights)
            self.preprocessing = default_preprocessing
        elif args.resnext50_32x4d or args.resnext101_32x8d:
            self.resnet50 =  ResNeXt(args,pretrained=True)
            self.preprocessing = default_torchvision
        elif args.bit:
            self.resnet50 = bit_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=1, non_head=True)
            self.resnet50.load_from(np.load("BiT-M-R50x1.npz"))
            self.preprocessing = empty_preprocessing
        else:
            self.resnet50 = resnet50.ResNet50(1)
            _init_backbone(self.resnet50, config.init_weights)
            self.preprocessing = default_preprocessing

        self.FPN = fpn.FPN(self.resnet50)
        self.RPN = RPN()
        if args.recursive:
            self.RCNN = RecursiveRCNN(args)
        else:
            self.RCNN = RCNN(args)
        self.args = args

    def forward(self, image, im_info, gt_boxes=None, isEval = False, extra = {}):
        image = image[:, :, :int(im_info[0, 0]), :int(im_info[0, 1])]
        image = self.preprocessing(image)
        image = pad_tensor_to_multiple_number(image, 64)
        if self.args.flip_aug:
            extra['aug_data'] = extra['aug_data'][:, :, :int(im_info[0, 0]), :int(im_info[0, 1])]
            extra['aug_data'] = self.preprocessing(extra['aug_data'])
            extra['aug_data'] = pad_tensor_to_multiple_number(extra['aug_data'], 64)
        if self.training or isEval:
            return self._forward_train(image, im_info, gt_boxes, isEval, extra)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes,  isEval = False, extra = {}):
        loss_dict = {}
        fpn_fms = self.FPN(image)
        

        ## stride: 64,32,16,8,4
        rpn_rois, rpn_rois_inds, loss_dict_rpn = \
            self.RPN(fpn_fms, im_info, gt_boxes , isEval)

            

        if self.args.flip_JSD_0g:
             with torch.no_grad():
                 flip_fms = [x.flip(-1) for x in self.FPN(image.flip(-1))]
        if self.args.flip_JSD:
            if self.args.flip_aug:
                flip_fms = [x.flip(-1) for x in self.FPN(extra['aug_data'].flip(-1))]
            else:
                flip_fms = [x.flip(-1) for x in self.FPN(image.flip(-1))]    
        if self.args.recursive:
            loss_dict_rcnn = self.RCNN(
                    fpn_fms,
                    rpn_rois, rpn_rois_inds, im_info,
                    gt_boxes = gt_boxes,
                    isEval = isEval, 
                    flip_fms = None if not self.args.flip_JSD else flip_fms, 
                    extra=extra)
        else:
            with torch.no_grad():
                rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
                    rpn_rois, rpn_rois_inds, im_info, gt_boxes, top_k=2)

            loss_dict_rcnn = self.RCNN(
                    fpn_fms, rcnn_rois, rcnn_labels, rcnn_bbox_targets,
                    isEval, 
                    flip_fms = None if not self.args.flip_JSD else flip_fms, 
                    extra=extra)
        

        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.FPN(image)
        rpn_rois, rpn_rois_inds = self.RPN(fpn_fms, im_info)
        
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        pred_bbox[:, :-1] /= im_info[0, 2]
        return pred_bbox.cpu().detach()

class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        rpn_channel = 256
        num_cell_anchors = config.num_cell_anchors
        self.rpn_conv = nn.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = nn.Conv2d(rpn_channel, num_cell_anchors, kernel_size=1, stride=1)
        self.rpn_bbox_offsets = nn.Conv2d(rpn_channel, num_cell_anchors * 4, kernel_size=1, stride=1)

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        self.anchor_scales = config.anchor_scales[::-1]

    def forward(self, features, im_info, boxes=None, eval = False):
        assert len(features) == config.num_anchor_scales
        # get the predict results
        pred_cls_score_list = []
        pred_bbox_offsets_list = []
        all_anchors_list = []
        for x in features:
            t = F.relu(self.rpn_conv(x))
            pred_cls_score_list.append(self.rpn_cls_score(t))
            pred_bbox_offsets_list.append(self.rpn_bbox_offsets(t))
        # get anchors
        max_fm_stride = 2 ** (len(features) + 1)
        fm_stride = max_fm_stride
        anchor_scales = config.anchor_scales[::-1]
        for i in range(config.num_anchor_scales):
            anchor_scale = [anchor_scales[i]]
            layer_anchors = generate_anchors_opr(features[i], fm_stride, anchor_scale,
                        config.anchor_aspect_ratios, config.anchor_base_size)
            fm_stride = fm_stride // 2
            all_anchors_list.append(layer_anchors)
        # sample from the predictions
        with torch.no_grad():
            rpn_rois, rpn_rois_inds = find_top_rpn_proposals(
                self.training or eval, pred_bbox_offsets_list, pred_cls_score_list,
                all_anchors_list, im_info)

        if self.training or eval:
            with torch.no_grad():
                rpn_labels, rpn_bbox_targets = fpn_anchor_target(
                    boxes, im_info, all_anchors_list)
            pred_cls_score, pred_bbox_offsets = fpn_rpn_reshape(
                pred_cls_score_list, pred_bbox_offsets_list)

            # rpn loss
            valid_masks = rpn_labels >= 0
            objectness_loss = F.binary_cross_entropy_with_logits(
                pred_cls_score[valid_masks],
                rpn_labels[valid_masks].to(torch.float32),
                reduction="sum")
            pos_masks = rpn_labels == 1
            localization_loss = smooth_l1_loss(
                pred_bbox_offsets[pos_masks],
                rpn_bbox_targets[pos_masks],
                config.rpn_smooth_l1_beta, reduction="sum")
            normalizer = 1.0 / (config.train_batch_per_gpu * config.num_sample_anchors)
            loss_rpn_cls = objectness_loss * normalizer  # cls: classification loss
            loss_rpn_loc = localization_loss * normalizer
            loss_dict = {}
            loss_dict['loss_rpn_cls'] = loss_rpn_cls
            loss_dict['loss_rpn_loc'] = loss_rpn_loc
            return rpn_rois, rpn_rois_inds, loss_dict
        else:
            return rpn_rois, rpn_rois_inds

def _KLD(P,Q):
    KLD = torch.sum(P * torch.log2(P/(Q + 1e-7)),dim=-1,keepdim=True)
    return KLD
def _flip_loss_JSD(A,B):
    A = A.view(A.size()[0],-1)
    B = B.view(B.size()[0],-1)
    M = (A + B) * .5
    JSD = (_KLD(A,M) + _KLD(B,M)) * .5
    return torch.mean(JSD)


def _diff_loss(feature0,feature1):
    f0_max = torch.max(feature0,dim=1,keepdim=True).values + 1e-5
    f1_max = torch.max(feature1,dim=1,keepdim=True).values + 1e-5
    f0_norm = feature0 / f0_max
    f1_norm = feature1 / f1_max
    L1 = torch.abs(f0_norm-f1_norm)
    return 1 - torch.mean(torch.mean(L1,dim=1))
class RCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # pooler
        stride = [4, 8, 16, 32]
        pooler_scales = tuple(1.0 / k for k in stride)
        
        
        self.box_pooler = ROIPooler(
            output_size=7,
            scales=pooler_scales,
            sampling_ratio=0,
            pooler_type='ROIAlignV2',
        )
        # roi head
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1044, 1024)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        # box predictor
        self.emd_pred_cls_0 = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta_0 = nn.Linear(1024, (config.num_classes - 1) * 4)
        self.emd_pred_cls_1 = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta_1 = nn.Linear(1024, (config.num_classes - 1) * 4)
        
        self.ref_pred_cls_0 = nn.Linear(1024, config.num_classes)
        self.ref_pred_delta_0 = nn.Linear(1024, (config.num_classes - 1) * 4)
        self.ref_pred_cls_1 = nn.Linear(1024, config.num_classes)
        self.ref_pred_delta_1 = nn.Linear(1024, (config.num_classes - 1) * 4)
        for l in [self.emd_pred_cls_0, self.emd_pred_cls_1,
                self.ref_pred_cls_0, self.ref_pred_cls_1]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        for l in [self.emd_pred_delta_0, self.emd_pred_delta_1,
                self.ref_pred_delta_0, self.ref_pred_delta_1]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)
    def _half_forward(self, fpn_fms, rcnn_rois, keep_pool_feature = False):
         # input p2-p5
        fpn_fms = fpn_fms[1:][::-1]
        
        
        
        pool_features = self.box_pooler(fpn_fms, rcnn_rois)
        
        
        flatten_feature = torch.flatten(pool_features, start_dim=1)

        flatten_feature = F.relu(self.fc1(flatten_feature))
        flatten_feature = F.relu(self.fc2(flatten_feature))

        pred_emd_pred_cls_0 = self.emd_pred_cls_0(flatten_feature)
        pred_emd_pred_delta_0 = self.emd_pred_delta_0(flatten_feature)
        
        pred_emd_pred_cls_1 = self.emd_pred_cls_1(flatten_feature)
        pred_emd_pred_delta_1 = self.emd_pred_delta_1(flatten_feature)
        pred_emd_scores_0 = F.softmax(pred_emd_pred_cls_0, dim=-1)
        pred_emd_scores_1 = F.softmax(pred_emd_pred_cls_1, dim=-1)
        
        # cons refine feature
        boxes_feature_0 = torch.cat((pred_emd_pred_delta_0,
            pred_emd_scores_0[:, 1][:, None]), dim=1).repeat(1, 4)
        boxes_feature_1 = torch.cat((pred_emd_pred_delta_1,
            pred_emd_scores_1[:, 1][:, None]), dim=1).repeat(1, 4)

        boxes_feature_0 = torch.cat((flatten_feature, boxes_feature_0), dim=1)
        boxes_feature_1 = torch.cat((flatten_feature, boxes_feature_1), dim=1)
        refine_feature_0 = F.relu(self.fc3(boxes_feature_0))
        refine_feature_1 = F.relu(self.fc3(boxes_feature_1))
        # refine
        pred_ref_pred_cls_0 = self.ref_pred_cls_0(refine_feature_0)
        pred_ref_pred_delta_0 = self.ref_pred_delta_0(refine_feature_0)
        pred_ref_pred_cls_1 = self.ref_pred_cls_1(refine_feature_1)
        pred_ref_pred_delta_1 = self.ref_pred_delta_1(refine_feature_1)

        results = [pred_emd_pred_cls_0, pred_emd_pred_delta_0, pred_emd_pred_cls_1, pred_emd_pred_delta_1,
            pred_ref_pred_cls_0, pred_ref_pred_delta_0, pred_ref_pred_cls_1, pred_ref_pred_delta_1]
        if keep_pool_feature:
            return results + [pool_features,(refine_feature_0,refine_feature_1)]

        return results
    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None, eval = False, flip_fms = None, extra = {}):
        # input p2-p5
        pred_emd_pred_cls_0, pred_emd_pred_delta_0, pred_emd_pred_cls_1, pred_emd_pred_delta_1,\
            pred_ref_pred_cls_0, pred_ref_pred_delta_0, pred_ref_pred_cls_1, pred_ref_pred_delta_1,\
                pool_features,refine_features = self._half_forward(fpn_fms, rcnn_rois, keep_pool_feature = True)
        

        if self.training or eval:
            loss0 = emd_loss(
                        pred_emd_pred_delta_0, pred_emd_pred_cls_0,
                        pred_emd_pred_delta_1, pred_emd_pred_cls_1,
                        bbox_targets, labels)
            loss1 = emd_loss(
                        pred_emd_pred_delta_1, pred_emd_pred_cls_1,
                        pred_emd_pred_delta_0, pred_emd_pred_cls_0,
                        bbox_targets, labels)

            loss2 = emd_loss(
                        pred_ref_pred_delta_0, pred_ref_pred_cls_0,
                        pred_ref_pred_delta_1, pred_ref_pred_cls_1,
                        bbox_targets, labels)
            loss3 = emd_loss(
                        pred_ref_pred_delta_1, pred_ref_pred_cls_1,
                        pred_ref_pred_delta_0, pred_ref_pred_cls_0,
                        bbox_targets, labels)
                        
            loss_rcnn = torch.cat([loss0, loss1], axis=1)
            loss_ref = torch.cat([loss2, loss3], axis=1)
            with torch.no_grad():
                _, min_indices_rcnn = loss_rcnn.min(axis=1)
                _, min_indices_ref = loss_ref.min(axis=1)
            loss_rcnn = loss_rcnn[torch.arange(loss_rcnn.shape[0]), min_indices_rcnn]
            loss_rcnn = loss_rcnn.sum()/loss_rcnn.shape[0]
            loss_ref = loss_ref[torch.arange(loss_ref.shape[0]), min_indices_ref]
            loss_ref = loss_ref.sum()/loss_ref.shape[0]
            #loss, _ = loss.min(axis=1)
            #loss_emd = loss.sum()/loss.shape[0]
            
            loss_dict = {}
            loss_dict['loss_rcnn_emd'] = loss_rcnn
            loss_dict['loss_ref_emd'] = loss_ref

            if self.args.flip_JSD:
                if self.args.flip_JSD_0g:
                    with torch.no_grad():
                        _, _, _, _,\
                        f_pred_ref_pred_cls_0, _, \
                            f_pred_ref_pred_cls_1, _ = self._half_forward(flip_fms, rcnn_rois)
                else:
                    _, _, _, _,\
                        f_pred_ref_pred_cls_0, _, \
                            f_pred_ref_pred_cls_1, _ = self._half_forward(flip_fms, rcnn_rois)
                loss_flip_JSD = _flip_loss_JSD(F.softmax(pred_ref_pred_cls_0, dim=-1),F.softmax(f_pred_ref_pred_cls_0, dim=-1))
                loss_flip_JSD += _flip_loss_JSD(F.softmax(pred_ref_pred_cls_1, dim=-1),F.softmax(f_pred_ref_pred_cls_1, dim=-1))
                loss_dict['loss_flip_JSD'] = loss_flip_JSD 
         
            if self.args.diff_loss:   
                loss_dict['diff_loss'] =  _diff_loss(refine_features[0],refine_features[1])     


            return loss_dict
        else:
            pred_ref_scores_0 = F.softmax(pred_ref_pred_cls_0, dim=-1)
            pred_ref_scores_1 = F.softmax(pred_ref_pred_cls_1, dim=-1)

            pred_bbox_0 = restore_bbox(rcnn_rois[:, 1:5], pred_ref_pred_delta_0, True)
            pred_bbox_1 = restore_bbox(rcnn_rois[:, 1:5], pred_ref_pred_delta_1, True)
            pred_bbox_0 = torch.cat([pred_bbox_0, pred_ref_scores_0[:, 1].reshape(-1,1)], dim=1)
            pred_bbox_1 = torch.cat([pred_bbox_1, pred_ref_scores_1[:, 1].reshape(-1,1)], dim=1)
            pred_bbox = torch.cat((pred_bbox_0, pred_bbox_1), dim=1).reshape(-1,5)
            return pred_bbox

def emd_loss(p_b0, p_c0, p_b1, p_c1, targets, labels):
    pred_box = torch.cat([p_b0, p_b1], dim=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_c0, p_c1], dim=1).reshape(-1, p_c0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.flatten()
    # loss for regression
    fg_inds = torch.nonzero(labels>0).flatten()
    loss_box_reg = smooth_l1_loss(
        pred_box[fg_inds],
        targets[fg_inds],
        config.rcnn_smooth_l1_beta, 'none').sum(axis=1)
    # loss for classification
    loss_cls = F.cross_entropy(pred_score, labels.long(), reduction='none', ignore_index=-1)
    loss = loss_cls
    loss[fg_inds] = loss[fg_inds] + loss_box_reg
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)



def emd_loss_multi(p_b_list, p_c_list, targets, labels, top_k):
    #print([x.size() for x in p_b_list])
    pred_box = torch.cat(p_b_list, dim=1).reshape(-1, p_b_list[0].shape[-1]) 
    pred_score = torch.cat(p_c_list, dim=1).reshape(-1, p_c_list[0].shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.flatten()
    # loss for regression
    fg_inds = torch.nonzero(labels>0).flatten()
    loss_box_reg = smooth_l1_loss(
        pred_box[fg_inds],
        targets[fg_inds],
        config.rcnn_smooth_l1_beta, 'none').sum(axis=1)
    # loss for classification
    loss_cls = F.cross_entropy(pred_score, labels.long(), reduction='none', ignore_index=-1)
    loss = loss_cls
    loss[fg_inds] = loss[fg_inds] + loss_box_reg 
    loss = loss.reshape(-1, top_k).sum(axis=1)
    return loss.reshape(-1, 1)

class RecursiveRCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # pooler
        stride = [4, 8, 16, 32]
        pooler_scales = tuple(1.0 / k for k in stride)
        self.box_pooler = ROIPooler(
            output_size=7,
            scales=pooler_scales,
            sampling_ratio=0,
            pooler_type='ROIAlignV2',
        )
        # roi head
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1044, 1024)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        # box predictor
        self.emd_pred_cls = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta = nn.Linear(1024, (config.num_classes - 1) * 4)
        
        self.ref_pred_cls = nn.Linear(1024, config.num_classes)
        self.ref_pred_delta = nn.Linear(1024, (config.num_classes - 1) * 4)

        #self.controller = nn.Linear(1024, 1)
        self.controller = nn.Linear(1024, 1024)


        for l in [self.emd_pred_cls, self.ref_pred_cls, self.controller]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        for l in [self.emd_pred_delta,self.ref_pred_delta]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def _recursive_function(self, flatten_feature, pre_feature = None):
        if pre_feature is None:
            pred_emd_pred_cls = self.emd_pred_cls(flatten_feature)
            pred_emd_pred_delta = self.emd_pred_delta(flatten_feature)
        else:
            pre_feature = self.controller(pre_feature)
            pred_emd_pred_cls = self.emd_pred_cls(flatten_feature + pre_feature)
            pred_emd_pred_delta = self.emd_pred_delta(flatten_feature + pre_feature)

        
        pred_emd_scores = F.softmax(pred_emd_pred_cls, dim=-1)
        
        # cons refine feature
        boxes_feature = torch.cat((pred_emd_pred_delta,
            pred_emd_scores[:, 1][:, None]), dim=1).repeat(1, 4)

        boxes_feature = torch.cat((flatten_feature, boxes_feature), dim=1)
        refine_feature = F.relu(self.fc3(boxes_feature))
        # refine
        pred_ref_pred_cls = self.ref_pred_cls(refine_feature)
        pred_ref_pred_delta = self.ref_pred_delta(refine_feature)

        return refine_feature, pred_ref_pred_cls, pred_ref_pred_delta, pred_emd_pred_cls, pred_emd_pred_delta

    def _recursive_forward(self, fpn_fms, rcnn_rois, keep_pool_feature = False):
         # input p2-p5
        fpn_fms = fpn_fms[1:][::-1]

        pool_features = self.box_pooler(fpn_fms, rcnn_rois)
        flatten_feature = torch.flatten(pool_features, start_dim=1)
        flatten_feature = F.relu(self.fc1(flatten_feature))
        flatten_feature = F.relu(self.fc2(flatten_feature))

        refine_feature = None
        pred_cls_unrefined = []
        pred_delta_unrefined = []
        pred_cls_ = []
        pred_delta_ = []
        for indx in range(2):
            refine_feature,\
            pred_cls,\
            pred_delta, pred_emd_pred_cls, pred_emd_pred_delta = self._recursive_function(flatten_feature, refine_feature)
            pred_cls_.append(pred_cls)
            pred_delta_.append(pred_delta)
            pred_cls_unrefined.append(pred_emd_pred_cls)
            pred_delta_unrefined.append(pred_emd_pred_delta)

        if keep_pool_feature:
            return pred_cls_, pred_delta_, pred_cls_unrefined, pred_delta_unrefined,  pool_features

        return pred_cls_, pred_delta_, pred_cls_unrefined, pred_delta_unrefined
    
    def forward(self, fpn_fms,rpn_rois, rpn_rois_inds = None, im_info = None, gt_boxes=None, isEval = False, flip_fms = None, extra = {}):
        
        if self.training or isEval:
            with torch.no_grad():
                rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
                    rpn_rois, rpn_rois_inds, im_info, gt_boxes, top_k=2)
        else:
            rcnn_rois = rpn_rois

        pred_ref_pred_cls, pred_ref_pred_delta, pred_cls_unrefined, pred_delta_unrefined, \
        pool_features = self._recursive_forward(fpn_fms, rcnn_rois, keep_pool_feature = True)
        

        if self.training or isEval:

            #loss_rcnn = emd_loss_multi(pred_delta_unrefined, pred_cls_unrefined,rcnn_bbox_targets,rcnn_labels,top_k=2)
            #loss_ref = emd_loss_multi(pred_ref_pred_delta, pred_ref_pred_cls,rcnn_bbox_targets,rcnn_labels,top_k=2)

            loss0 = emd_loss(
                        pred_delta_unrefined[0], pred_cls_unrefined[0],
                        pred_delta_unrefined[1], pred_cls_unrefined[1],
                        rcnn_bbox_targets, rcnn_labels)
            loss1 = emd_loss(
                        pred_delta_unrefined[1], pred_cls_unrefined[1],
                        pred_delta_unrefined[0], pred_cls_unrefined[0],
                        rcnn_bbox_targets, rcnn_labels)

            loss2 = emd_loss(
                        pred_ref_pred_delta[0], pred_ref_pred_cls[0],
                        pred_ref_pred_delta[1], pred_ref_pred_cls[1],
                        rcnn_bbox_targets, rcnn_labels)
            loss3 = emd_loss(
                        pred_ref_pred_delta[1], pred_ref_pred_cls[1],
                        pred_ref_pred_delta[0], pred_ref_pred_cls[0],
                        rcnn_bbox_targets, rcnn_labels)
                        
            loss_rcnn = torch.cat([loss0, loss1], axis=1)
            loss_ref = torch.cat([loss2, loss3], axis=1)
            loss_rcnn = torch.cat([loss0, loss1], axis=1)
            loss_ref = torch.cat([loss2, loss3], axis=1)


            with torch.no_grad():
                _, min_indices_rcnn = loss_rcnn.min(axis=1)
                _, min_indices_ref = loss_ref.min(axis=1)
            loss_rcnn = loss_rcnn[torch.arange(loss_rcnn.shape[0]), min_indices_rcnn]
            loss_rcnn = loss_rcnn.sum()/loss_rcnn.shape[0]
            loss_ref = loss_ref[torch.arange(loss_ref.shape[0]), min_indices_ref]
            loss_ref = loss_ref.sum()/loss_ref.shape[0]


            loss_dict = {}
            loss_dict['loss_rcnn_emd'] = loss_rcnn
            loss_dict['loss_ref_emd'] = loss_ref

            if self.args.flip_JSD:
                if self.args.flip_JSD_0g:
                    with torch.no_grad():
                        f_pred_ref_pred_cls, _, pred_cls_unrefined, _ = self._recursive_forward(flip_fms, rcnn_rois)
                else:
                    f_pred_ref_pred_cls, _ = self._recursive_forward(flip_fms, rcnn_rois)
                loss_flip_JSD = _flip_loss_JSD(F.softmax(pred_ref_pred_cls[0], dim=-1),F.softmax(f_pred_ref_pred_cls[0], dim=-1))
                loss_flip_JSD += _flip_loss_JSD(F.softmax(pred_ref_pred_cls[1], dim=-1),F.softmax(f_pred_ref_pred_cls[1], dim=-1))
                loss_dict['loss_flip_JSD'] = loss_flip_JSD
                
               
            return loss_dict
        else:
            pred_bboxes = None
            for p_cls,p_delta in zip(pred_ref_pred_cls,pred_ref_pred_delta):
                pred_ref_scores = F.softmax(p_cls, dim=-1)
                pred_bbox = restore_bbox(rcnn_rois[:, 1:5], p_delta, True)
                if pred_bboxes is None:
                    pred_bboxes = torch.cat([pred_bbox, pred_ref_scores[:, 1].reshape(-1,1)], dim=1)
                else:
                    pred_bbox = torch.cat([pred_bbox, pred_ref_scores[:, 1].reshape(-1,1)], dim=1)
                    
                    pred_bboxes = torch.cat([pred_bboxes, pred_bbox], dim=0)
            
            #pred_bbox = torch.cat((pred_bbox_0, pred_bbox_1), dim=1).reshape(-1,5)
            return pred_bboxes

def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]
            ).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]
            ).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox

def _init_weights(layer):
    nn.init.kaiming_uniform_(layer.weight, a=1)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

if __name__ == "__main__":
    net = Network()
    data = torch.Tensor(np.random.rand(2, 3, 800, 800))
    box = torch.Tensor(np.random.rand(2, 8, 5)*100)
    box[:, :, 2:4]+= box[:, :, :2]
    box[:, :-1, 4] = 1
    box[:, -2, 4] = -1
    box[:, -1, 4] = -1
    box[0,0,0] = 440
    box[0,0,1] = 440
    box[0,0,2] = 930
    box[0,0,3] = 930
    im_info = torch.tensor([[800,800,0,0,0,8],[800,800,0,0,0,8]]).cuda()
    net.cuda()
    data = data.cuda()
    box = box.cuda()
    net.train()
    output = net(data, im_info, box)
    print(output)
