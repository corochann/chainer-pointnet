#!/usr/bin/env python

from __future__ import print_function
import argparse
from distutils.util import strtobool
import os

import chainer
from chainer import serializers
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.dataset import to_device
from chainer.datasets import TransformDataset, SubDataset
from chainer.training import extensions as E

from chainer_pointnet.models.kdnet.kdnet_cls import KDNetCls
from chainer_pointnet.models.pointnet.pointnet_cls import PointNetCls
from chainer_pointnet.models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG
from chainer_pointnet.models.pointnet2.pointnet2_cls_ssg import PointNet2ClsSSG

from ply_dataset import get_train_dataset, get_test_dataset

from chainer_pointnet.utils.kdtree import calc_max_level


def main():
    parser = argparse.ArgumentParser(
        description='ModelNet40 classification')
    # parser.add_argument('--conv-layers', '-c', type=int, default=4)
    parser.add_argument('--method', '-m', type=str, default='point_cls')
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--dropout_ratio', type=float, default=0.3)
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--epoch', '-e', type=int, default=250)
    # parser.add_argument('--unit-num', '-u', type=int, default=16)
    parser.add_argument('--seed', '-s', type=int, default=777)
    parser.add_argument('--protocol', type=int, default=2)
    parser.add_argument('--model_filename', type=str, default='model.npz')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--trans', type=strtobool, default='true')
    parser.add_argument('--use_bn', type=strtobool, default='true')
    args = parser.parse_args()

    seed = args.seed
    method = args.method
    num_point = args.num_point
    num_class = 40
    debug = False

    # Dataset preparation
    train = get_train_dataset(num_point=num_point)
    val = get_test_dataset(num_point=num_point)
    if method == 'kdnet_cls':
        from chainer_pointnet.utils.kdtree import TransformKDTreeCls
        max_level = calc_max_level(num_point)
        train = TransformDataset(train, TransformKDTreeCls(max_level=max_level))
        val = TransformDataset(val, TransformKDTreeCls(max_level=max_level))
        points, split_dims, t = train[0]
        print('converted to kdtree dataset train', points.shape, split_dims.shape, t)
        points, split_dims, t = val[0]
        print('converted to kdtree dataset val', points.shape, split_dims.shape, t)

    if debug:
        # use few train dataset
        train = SubDataset(train, 0, 50)

    # Network
    # n_unit = args.unit_num
    # conv_layers = args.conv_layers
    trans = args.trans
    use_bn = args.use_bn
    dropout_ratio = args.dropout_ratio
    from chainer.dataset.convert import concat_examples
    converter = concat_examples

    if method == 'point_cls':
        print('Train PointNetCls model... trans={} use_bn={} dropout={}'
              .format(trans, use_bn, dropout_ratio))
        model = PointNetCls(
            out_dim=num_class, in_dim=3, middle_dim=64, dropout_ratio=dropout_ratio,
            trans=trans, trans_lam1=0.001, trans_lam2=0.001, use_bn=use_bn)
    elif method == 'point2_cls_ssg':
        print('Train PointNet2ClsSSG model... use_bn={} dropout={}'
              .format(use_bn, dropout_ratio))
        model = PointNet2ClsSSG(
            out_dim=num_class, in_dim=3,
            dropout_ratio=dropout_ratio, use_bn=use_bn)
    elif method == 'point2_cls_msg':
        print('Train PointNet2ClsMSG model... use_bn={} dropout={}'
              .format(use_bn, dropout_ratio))
        model = PointNet2ClsMSG(
            out_dim=num_class, in_dim=3,
            dropout_ratio=dropout_ratio, use_bn=use_bn)
    elif method == 'kdnet_cls':
        print('Train KdNetCls model... use_bn={} dropout={}'
              .format(use_bn, dropout_ratio))
        model = KDNetCls(
            out_dim=num_class, in_dim=3,
            dropout_ratio=dropout_ratio)  # use_bn=use_bn

        def kdnet_converter(batch, device=None, padding=None):
            # concat_examples to CPU at first.
            result = concat_examples(batch, device=None, padding=padding)
            out_list = []
            for elem in result:
                if elem.dtype != object:
                    # Send to GPU for int/float dtype array.
                    out_list.append(to_device(device, elem))
                else:
                    # Do NOT send to GPU for dtype=object array.
                    out_list.append(elem)
            return tuple(out_list)

        converter = kdnet_converter
    else:
        raise ValueError('[ERROR] Invalid method {}'.format(method))

    train_iter = iterators.SerialIterator(train, args.batchsize)
    val_iter = iterators.SerialIterator(
        val, args.batchsize, repeat=False, shuffle=False)

    device = args.gpu
    # classifier = Classifier(model, device=device)
    classifier = model
    load_model = False
    if load_model:
        serializers.load_npz(
            os.path.join(args.out, args.model_filename), classifier)
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        classifier.to_gpu()  # Copy the model to the GPU

    optimizer = optimizers.Adam()
    optimizer.setup(classifier)

    updater = training.StandardUpdater(
        train_iter, optimizer, converter=converter, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    from chainerex.training.extensions import schedule_optimizer_value
    from chainer.training.extensions import observe_value
    # trainer.extend(observe_lr)
    observation_key = 'lr'
    trainer.extend(observe_value(
        observation_key,
        lambda trainer: trainer.updater.get_optimizer('main').alpha))
    trainer.extend(schedule_optimizer_value(
        [10, 20, 100, 150, 200, 230],
        [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]))

    trainer.extend(E.Evaluator(
        val_iter, classifier, converter=converter, device=args.gpu))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(
        ['epoch', 'main/loss', 'main/cls_loss', 'main/trans_loss1',
         'main/trans_loss2', 'main/accuracy', 'validation/main/loss',
         # 'validation/main/cls_loss',
         # 'validation/main/trans_loss1', 'validation/main/trans_loss2',
         'validation/main/accuracy', 'lr', 'elapsed_time']))
    trainer.extend(E.ProgressBar(update_interval=10))

    if args.resume:
        serializers.load_npz(args.resume, trainer)
    trainer.run()

    # --- save classifier ---
    # protocol = args.protocol
    # classifier.save_pickle(
    #     os.path.join(args.out, args.model_filename), protocol=protocol)
    serializers.save_npz(
        os.path.join(args.out, args.model_filename), classifier)


if __name__ == '__main__':
    main()
