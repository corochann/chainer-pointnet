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
from chainer.dataset import to_device, concat_examples
from chainer.datasets import TransformDataset
from chainer.training import extensions as E

from chainer_pointnet.models.kdcontextnet.kdcontextnet_seg import \
    KDContextNetSeg
from chainer_pointnet.models.kdnet.kdnet_seg import KDNetSeg
from chainer_pointnet.models.pointnet.pointnet_seg import PointNetSeg
from chainer_pointnet.models.pointnet2.pointnet2_seg_ssg import PointNet2SegSSG

from s3dis_dataset import get_dataset

from chainer_pointnet.utils.kdtree import calc_max_level, TransformKDTreeSeg


def main():
    parser = argparse.ArgumentParser(
        description='S3DIS segmentation')
    parser.add_argument('--method', '-m', type=str, default='point_seg')
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--dropout_ratio', type=float, default=0.0)
    parser.add_argument('--num_point', type=int, default=4096)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--epoch', '-e', type=int, default=250)
    parser.add_argument('--seed', '-s', type=int, default=777)
    parser.add_argument('--protocol', type=int, default=2)
    parser.add_argument('--model_filename', type=str, default='model.npz')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--trans', type=strtobool, default='false')
    parser.add_argument('--use_bn', type=strtobool, default='true')
    args = parser.parse_args()

    seed = args.seed
    out_dir = args.out
    method = args.method
    num_point = args.num_point

    try:
        os.makedirs(out_dir, exist_ok=True)
        import chainerex.utils as cl
        fp = os.path.join(out_dir, 'args.json')
        cl.save_json(fp, vars(args))
        print('save args to', fp)
    except ImportError:
        pass

    # S3DIS dataset has 13 labels
    num_class = 13
    in_dim = 9

    # Dataset preparation
    train, val = get_dataset(num_point=num_point)
    if method == 'kdnet_seg' or method == 'kdcontextnet_seg':
        from chainer_pointnet.utils.kdtree import TransformKDTreeSeg, \
            calc_max_level
        max_level = calc_max_level(num_point)
        print('kdnet max_level {}'.format(max_level))
        return_split_dims = (method == 'kdnet_seg')
        train = TransformDataset(train, TransformKDTreeSeg(
            max_level=max_level, return_split_dims=return_split_dims))
        val = TransformDataset(val, TransformKDTreeSeg(
            max_level=max_level, return_split_dims=return_split_dims))
        if method == 'kdnet_seg':
            # Debug print
            points, split_dims, t = train[0]
            print('converted to kdnet dataset train', points.shape, split_dims.shape, t.shape)
            points, split_dims, t = val[0]
            print('converted to kdnet dataset val', points.shape, split_dims.shape, t.shape)
        if method == 'kdcontextnet_seg':
            # Debug print
            points, t = train[0]
            print('converted to kdcontextnet dataset train', points.shape, t.shape)
            points, t = val[0]
            print('converted to kdcontextnet dataset val', points.shape, t.shape)

    # Network
    trans = args.trans
    use_bn = args.use_bn
    dropout_ratio = args.dropout_ratio
    converter = concat_examples
    if method == 'point_seg':
        print('Train PointNetSeg model... trans={} use_bn={} dropout={}'
              .format(trans, use_bn, dropout_ratio))
        model = PointNetSeg(
            out_dim=num_class, in_dim=in_dim, middle_dim=64, dropout_ratio=dropout_ratio,
            trans=trans, trans_lam1=0.001, trans_lam2=0.001, use_bn=use_bn)
    elif method == 'point2_seg_ssg':
        print('Train PointNet2SegSSG model... use_bn={} dropout={}'
              .format(use_bn, dropout_ratio))
        model = PointNet2SegSSG(
            out_dim=num_class, in_dim=in_dim,
            dropout_ratio=dropout_ratio, use_bn=use_bn)
    elif method == 'kdnet_seg':
        print('Train KDNetSeg model... use_bn={} dropout={}'
              .format(use_bn, dropout_ratio))
        model = KDNetSeg(
            out_dim=num_class, in_dim=in_dim,
            dropout_ratio=dropout_ratio, use_bn=use_bn, max_level=max_level)

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
    elif method == 'kdcontextnet_seg':
        print('Train KDContextNetSeg model... use_bn={} dropout={}'
              .format(use_bn, dropout_ratio))
        model = KDContextNetSeg(
            out_dim=num_class, in_dim=in_dim,
            dropout_ratio=dropout_ratio, use_bn=use_bn)
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
        train_iter, optimizer, device=args.gpu, converter=converter)

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
        val_iter, classifier, device=args.gpu, converter=converter))
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
