# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2


import torch
from torch.autograd import Variable
import torch.nn as nn
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, sampler, coral

from model.utils.parser_func import parse_args, set_dataset_args

if __name__ == '__main__':

    torch.manual_seed(100)
    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    aux_label = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        aux_label = aux_label.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    aux_label = Variable(aux_label)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.faster_rcnn.vgg16 import vgg16
    from model.faster_rcnn.resnet import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]


    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        checkpoint = torch.load(args.load_name)
        fasterRCNN.load_state_dict(checkpoint['model'])
        print("loaded checkpoint %s" % (args.load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(10000 / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    count_iter = 0
    flag = 0
    loss_avg = []
    avg_count = 0
    min_count=0

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        fasterRCNN.train()
        loss_temp = 0
        loss_temp_adv = 0
        start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)
        for step in range(iters_per_epoch):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)

            count_iter += 1

            im_data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.resize_(data_s[3].size()).copy_(data_s[3])

            rois_label, rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls_1, RCNN_loss_bbox_1, \
            RCNN_loss_cls_2, RCNN_loss_bbox_2 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=0)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls_1.mean() + RCNN_loss_bbox_1.mean() \
                   + RCNN_loss_cls_2.mean() + RCNN_loss_bbox_2.mean()

            loss_temp += loss.item()

            optimizer.zero_grad()
            loss.backward()
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], max_norm=20)
            optimizer.step()

            for p in fasterRCNN.RCNN_base.parameters():
                p.requires_grad = False
            for p in fasterRCNN.RCNN_rpn.parameters():
                p.requires_grad = False


            if flag == 0:
                rois_label, rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls_1, RCNN_loss_bbox_1, \
                RCNN_loss_cls_2, RCNN_loss_bbox_2 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=0)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                       + RCNN_loss_cls_1.mean() + RCNN_loss_bbox_1.mean() \
                       + RCNN_loss_cls_2.mean() + RCNN_loss_bbox_2.mean()

                im_data.resize_(data_t[0].size()).copy_(data_t[0])
                im_info.resize_(data_t[1].size()).copy_(data_t[1])
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                # gt_boxes.resize_(data_t[2].size()).copy_(data_t[2])
                # num_boxes.resize_(data_t[3].size()).copy_(data_t[3])

                adv_cls_loss, adv_bbox_loss = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=1)

                loss -= (adv_cls_loss + adv_bbox_loss) * 0.1

                optimizer.zero_grad()
                loss.backward()
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_norm=20)
                optimizer.step()
                print('{} minimum iters before maximum iter'.format(min_count))
                min_count = 0
                flag = 1


            for p in fasterRCNN.RCNN_base[2:].parameters():
                p.requires_grad = True
            for p in fasterRCNN.RCNN_rpn.parameters():
                p.requires_grad = True
            for p in fasterRCNN.RCNN_top_1.parameters():
                p.requires_grad = False
            for p in fasterRCNN.RCNN_top_2.parameters():
                p.requires_grad = False
            for p in fasterRCNN.RCNN_bbox_pred_1.parameters():
                p.requires_grad = False
            for p in fasterRCNN.RCNN_bbox_pred_2.parameters():
                p.requires_grad = False
            for p in fasterRCNN.RCNN_cls_score_1.parameters():
                p.requires_grad = False
            for p in fasterRCNN.RCNN_cls_score_2.parameters():
                p.requires_grad = False


            if flag == 1:
                min_count += 1
                adv_cls_loss, adv_bbox_loss = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=1)

                loss = (adv_cls_loss + adv_bbox_loss) * 0.1

                def get_average(list):
                    sum = 0.0
                    for item in list:
                        sum += item
                    return sum / len(list)

                loss_avg.append(loss.item())
                if len(loss_avg) > 20:
                    mean_before = get_average(loss_avg[:-1])
                    mean_after = get_average(loss_avg[1:])
                    if mean_after <= mean_before:
                        avg_count += 1
                        if avg_count == 10:
                            flag = 0
                            avg_count = 0
                    else:
                        avg_count = 0
                    loss_avg.pop(0)


                optimizer.zero_grad()
                loss.backward()
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_norm=20)
                optimizer.step()

                loss_temp_adv += loss.item()

            # for group in optimizer.param_groups:
            #     group['lr'] = lr_before

            for p in fasterRCNN.parameters():
                p.requires_grad = True
            for p in fasterRCNN.RCNN_base[0].parameters():
                p.requires_grad = False
            for p in fasterRCNN.RCNN_base[1].parameters():
                p.requires_grad = False

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                    loss_temp_adv /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls_1 = RCNN_loss_cls_1.mean().item()
                    loss_rcnn_box_1 = RCNN_loss_bbox_1.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls_1 = RCNN_loss_cls_1.item()
                    loss_rcnn_box_1 = RCNN_loss_bbox_1.item()
                    loss_rcnn_cls_2 = RCNN_loss_cls_2.item()
                    loss_rcnn_box_2 = RCNN_loss_bbox_2.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                    adv_cls_loss = adv_cls_loss.item()
                    adv_bbox_loss = adv_bbox_loss.item()

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, loss adv: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, loss_temp_adv, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls_1: %.4f, rcnn_box_1 %.4f, rcnn_cls_2 %.4f, rcnn_box_2 %.4f, adv cls loss %.4f, adv bbox loss %.4f" \
                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls_1, loss_rcnn_box_1, loss_rcnn_cls_2, loss_rcnn_box_2, adv_cls_loss, adv_bbox_loss))

                # print("\t\t\ttime cost: %f" % (end - start))
                # print(
                #     "\t\t\tadv cls loss %.4f, adv bbox loss %.4f" \
                #     % (adv_cls_loss, adv_bbox_loss))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_adv': loss_temp_adv,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls_1': loss_rcnn_cls_1,
                        'loss_rcnn_box_1': loss_rcnn_box_1,
                        'loss_rcnn_cls_2': loss_rcnn_cls_2,
                        'loss_rcnn_box_2': loss_rcnn_box_2,
                        'adv_cls_loss':adv_cls_loss,
                        'adv_bbox_loss':adv_bbox_loss
                    }
                    logger.add_scalars("logs_s_{}/mcd-PT".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                loss_temp_adv = 0
                start = time.time()

            if step % 1000 == 0:
                save_name = os.path.join(output_dir,
                                         'mcd_pure_target_{}_epoch_{}_step_{}.pth'.format(args.dataset, epoch, step))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                print('save model: {}'.format(save_name))
        save_name = os.path.join(output_dir,
                                 'mcd_pure_target_{}_epoch_{}.pth'.format(args.dataset, epoch))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
