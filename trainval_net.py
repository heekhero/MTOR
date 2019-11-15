# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pdb
import pprint
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, sampler, relation_matrix, guide_matrix, zero_params
from model.utils.parser_func import parse_args, set_dataset_args
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb

os.environ['CUDA_VISIBLE_DEVICES']='1'

if __name__ == '__main__':

    torch.manual_seed(100)
    args = parse_args()

    cfg.TRAIN.TEACHER_ROI_THRESHOLD = args.emma
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
                               imdb.num_classes, training=True, target=False)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True, target=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    s = torch.FloatTensor(1)
    t = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    aux_label = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        s = s.cuda()
        t = t.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        aux_label = aux_label.cuda()

    # make variable
    im_data = Variable(im_data)
    s = Variable(s)
    t = Variable(t)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    aux_label = Variable(aux_label)

    if args.cuda:
        cfg.CUDA = True

    from model.faster_rcnn.resnet_teacher import resnet_teacher
    from model.faster_rcnn.resnet_student import resnet_student

    if args.net == 'res101':
        teacher = resnet_teacher(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        student = resnet_student(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    teacher.create_architecture()
    student.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(student.named_parameters()).items():
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
        teacher = teacher.cuda()
        student = student.cuda()

    if args.resume:
        checkpoint = torch.load(args.load_name)
        student.load_state_dict(checkpoint['model'])
        teacher.load_state_dict(checkpoint['model'])
        print("loaded checkpoint %s" % (args.load_name))

    iters_per_epoch = int(10000 / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    count_iter = 0
    loss_temp = 0
    flag= False

    for p in teacher.parameters():
        p.requires_grad = False

    # teacher.apply(zero_params)

    modules = dict(teacher.named_modules())

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        student.train()
        teacher.train()
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
                data_t = next (data_iter_t)

            count_iter += 1

            im_data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.resize_(data_s[3].size()).copy_(data_s[3])

            rois_label, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox = student(im_data, im_info, gt_boxes, num_boxes)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()


            s.resize_(data_t[0].size()).copy_(data_t[0])
            t.resize_(data_t[1].size()).copy_(data_t[1])
            im_info.resize_(data_t[2].size()).copy_(data_t[2])
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()

            # if epoch > 2:
            rois_t, f_t, p_t = teacher(t, im_info, gt_boxes, num_boxes)
            if rois_t.size(1) != 0 and rois_t.size(1) < 100:
                f_s, p_s = student(s, im_info, gt_boxes, num_boxes, t_rois = rois_t)

                RCL = F.mse_loss(p_s, p_t.squeeze(0))

                emma_s = relation_matrix(f_s)
                emma_t = relation_matrix(f_t)

                EGL = F.mse_loss(emma_s, emma_t)

                M = guide_matrix(p_t)

                AGL = torch.sum((1 - emma_s) * M) / torch.sum(M)

                loss += RCL + EGL + AGL
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, module in dict(student.named_modules()).items():
                op_mod = modules[name]
                # if hasattr(op_mod, 'weight') and op_mod.weight is not None:
                #     op_mod.weight.data = 0.9 * op_mod.weight.data + 0.1 * module.weight.data
                if len(op_mod._parameters) != 0:
                    for k in op_mod._parameters:
                        if op_mod._parameters[k] is not None:
                            op_mod._parameters[k].data = args.eam * op_mod._parameters[k].data + (1 - args.eam) * \
                                                         module._parameters[k].data

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box,
                    }
                    logger.add_scalars("pf", info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

            if step % 500 == 0:
                save_name = os.path.join(output_dir, '{}_{}_.pth'.format(epoch, step))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': student.module.state_dict() if args.mGPUs else student.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                print('save model: {}'.format(save_name))

                os.environ['CUDA_VISIBLE_DEVICES']='6'
                my_env = os.environ.copy()
                my_env["PATH"] = "/data/fuminghao/anaconda3/envs/da_detc/bin:" + my_env["PATH"]
                child = subprocess.Popen(
                    ['python', 'test_net.py', '--dataset', 'cityscape_car', '--load_name',
                     save_name, '--fd', 'rand'],
                    env=my_env,
                    stdout=subprocess.PIPE
                )

    if args.use_tfboard:
        logger.close()
