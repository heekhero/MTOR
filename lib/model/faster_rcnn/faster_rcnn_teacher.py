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
from torchvision.ops import nms

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

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        base_feat = self.RCNN_base(im_data)

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        rois = Variable(rois)

        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        pooled_feat_post = self._head_to_tail(pooled_feat)

        bbox_pred = self.RCNN_bbox_pred(pooled_feat_post)
        cls_score = self.RCNN_cls_score(pooled_feat_post)
        cls_prob = F.softmax(cls_score, 1)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        post_inds = self.rois_filter(cls_prob, cfg.TRAIN.TEACHER_ROI_THRESHOLD)

        if len(post_inds) > 100:
            print(len(post_inds))
            post_inds = post_inds[torch.randperm(len(post_inds))[:100]]

        post_rois = rois[:, post_inds, :]
        post_cls_prob = cls_prob[:, post_inds, :]
        pooled_feat_post = pooled_feat_post[post_inds, :]

        # max_fg_prob, max_fg_inds = torch.max(post_cls_prob[0], dim=1)
        #
        # res_rois = None
        # res_cls_prob = None
        # res_feat = None
        # for cls_ind in range(1, post_cls_prob.size(2)):
        #     each_cls_inds = torch.nonzero((max_fg_inds==cls_ind).view(-1)).view(-1)
        #     each_max_fg_prob = max_fg_prob[each_cls_inds]
        #     keep = nms(post_rois[0, each_cls_inds, 1:], each_max_fg_prob, 0.7)
        #     if res_rois is None:
        #         res_rois = post_rois[:, keep, :]
        #         res_cls_prob = post_cls_prob[:, keep, :]
        #         res_feat = pooled_feat_post[keep, :]
        #     else:
        #         res_rois = torch.cat([res_rois, post_rois[:, keep, :]], dim=1)
        #         res_cls_prob = torch.cat([res_cls_prob, post_cls_prob[:, keep, :]], dim=1)
        #         res_fea = torch.cat([res_feat, pooled_feat_post[keep, :]], dim=0)

        if self.training:
            return post_rois, pooled_feat_post, post_cls_prob
        else:
            return rois, cls_prob, bbox_pred

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
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def rois_filter(self, cls_prob, threshold):
        max_prob, arg_max = torch.max(cls_prob.squeeze(0), dim=1)
        fg_inds = torch.nonzero((arg_max!=0).view(-1)).view(-1)
        max_prob = max_prob[fg_inds]

        return fg_inds[torch.nonzero((max_prob > threshold).view(-1)).view(-1)]
