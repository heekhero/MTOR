import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from torchvision.ops import RoIPool
from torchvision.ops import RoIAlign
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
import torch.nn.init as init
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_align = RoIAlign(output_size=(cfg.POOLING_SIZE, cfg.POOLING_SIZE), spatial_scale=1.0 / 16.0,
                                       sampling_ratio=2)
        self.RCNN_roi_pool = RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

    def forward(self, im_data, im_info, gt_boxes, num_boxes, target=0):
        batch_size = im_data.size(0)

        base_feat = self.RCNN_base(im_data)

        # feed base feature map to RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training and target == 0:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat_1 = self._head_to_tail(pooled_feat, tar=0)
        pooled_feat_2 = self._head_to_tail(pooled_feat, tar=1)


        # compute bbox offset
        bbox_pred_1 = self.RCNN_bbox_pred_1(pooled_feat_1)
        cls_score_1 = self.RCNN_cls_score_1(pooled_feat_1)
        cls_prob_1 = F.softmax(cls_score_1, 1)

        bbox_pred_2 = self.RCNN_bbox_pred_2(pooled_feat_2)
        cls_score_2 = self.RCNN_cls_score_2(pooled_feat_2)
        cls_prob_2 = F.softmax(cls_score_2, 1)

        if target == 1:
            adv_cls_loss = F.mse_loss(cls_score_1, cls_score_2)
            adv_bbox_loss = F.mse_loss(bbox_pred_1, bbox_pred_2)
            return adv_cls_loss, adv_bbox_loss

        if self.training and not self.class_agnostic and target == 0:
            # select the corresponding columns according to roi labels
            bbox_pred_view_1 = bbox_pred_1.view(bbox_pred_1.size(0), int(bbox_pred_1.size(1) / 4), 4)
            bbox_pred_select_1 = torch.gather(bbox_pred_view_1, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred_1 = bbox_pred_select_1.squeeze(1)

            bbox_pred_view_2 = bbox_pred_2.view(bbox_pred_2.size(0), int(bbox_pred_2.size(1) / 4), 4)
            bbox_pred_select_2 = torch.gather(bbox_pred_view_2, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred_2 = bbox_pred_select_2.squeeze(1)

        RCNN_loss_cls_1 = 0
        RCNN_loss_bbox_1 = 0

        RCNN_loss_cls_2 = 0
        RCNN_loss_bbox_2 = 0

        if self.training:
            RCNN_loss_cls_1 = F.cross_entropy(cls_score_1, rois_label)
            RCNN_loss_bbox_1 = _smooth_l1_loss(bbox_pred_1, rois_target, rois_inside_ws, rois_outside_ws)

            RCNN_loss_cls_2 = F.cross_entropy(cls_score_2, rois_label)
            RCNN_loss_bbox_2 = _smooth_l1_loss(bbox_pred_2, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob_1 = cls_prob_1.view(batch_size, rois.size(1), -1)
        bbox_pred_1 = bbox_pred_1.view(batch_size, rois.size(1), -1)


        if self.training:
            return rois_label, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls_1, RCNN_loss_bbox_1, RCNN_loss_cls_2, RCNN_loss_bbox_2
        else:
            return rois, cls_prob_1, bbox_pred_1, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls_1, RCNN_loss_bbox_1, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_1, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_2, 0, 0.02, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_2, 0, 0.002, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
