import numpy

import chainer
from chainer import functions, cuda
from chainer import links
from chainer import reporter

from chainer_pointnet.models.conv_block import ConvBlock
from chainer_pointnet.models.linear_block import LinearBlock
from chainer_pointnet.models.transform_net import TransformNet


def calc_trans_loss(t):
    # Loss to enforce the transformation as orthogonal matrix
    # t (batchsize, K, K) - transform matrix
    xp = cuda.get_array_module(t)
    bs, k1, k2 = t.shape
    assert k1 == k2
    mat_diff = functions.matmul(t, functions.transpose(t, (0, 2, 1)))
    mat_diff = mat_diff - xp.identity(k1, dtype=xp.float32)
    # divide by 2. is to make the behavior same with tf.
    # https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/nn/l2_loss
    return functions.sum(functions.batch_l2_norm_squared(mat_diff)) / 2.


class PointNetSeg(chainer.Chain):

    """Segmentation PointNet

    Input is (minibatch, K, N, 1), output is (minibatch, out_dim, N)

        Args:
            out_dim (int): output dimension, number of class for classification
            in_dim: input dimension for each point. default is 3, (x, y, z).
            middle_dim (int): hidden layer
            dropout_ratio (float): dropout ratio
            trans (bool): use TransformNet or not.
                False means not to use TransformNet, corresponds to
                PointNetVanilla. True corresponds to PointNet in the paper.
            trans_lam1 (float): regularization term for input transform.
                used in training. it is simply ignored when `trans` is False.
            trans_lam2 (float): regularization term for feature transform
                used in training. it is simply ignored when `trans` is False.
            compute_accuracy (bool): compute & report accuracy or not
    """

    def __init__(self, out_dim, in_dim=3, middle_dim=64, dropout_ratio=0.3,
                 use_bn=True, trans=True, trans_lam1=0.001, trans_lam2=0.001,
                 compute_accuracy=True):
        super(PointNetSeg, self).__init__()
        with self.init_scope():
            if trans:
                self.input_transform_net = TransformNet(k=in_dim, use_bn=use_bn)

            self.conv_block1 = ConvBlock(in_dim, 64, ksize=1, use_bn=use_bn)
            self.conv_block2 = ConvBlock(64, middle_dim, ksize=1, use_bn=use_bn)
            if trans:
                self.feature_transform_net = TransformNet(k=middle_dim, use_bn=use_bn)

            self.conv_block3 = ConvBlock(middle_dim, 64, ksize=1, use_bn=use_bn)
            self.conv_block4 = ConvBlock(64, 128, ksize=1, use_bn=use_bn)
            self.conv_block5 = ConvBlock(128, 1024, ksize=1, use_bn=use_bn)

            # --- concat point_feat & global_feat ---
            self.conv_block6 = ConvBlock(middle_dim + 1024, 512, ksize=1, use_bn=use_bn)
            self.conv_block7 = ConvBlock(512, 256, ksize=1, use_bn=use_bn)
            self.conv_block8 = ConvBlock(256, 128, ksize=1, use_bn=use_bn)
            self.conv_block9 = ConvBlock(128, 128, ksize=1, use_bn=use_bn)
            self.conv10 = links.Convolution2D(128, out_dim, ksize=1)

        self.in_dim = in_dim
        self.dropout_ratio = dropout_ratio
        self.trans = trans
        self.trans_lam1 = trans_lam1
        self.trans_lam2 = trans_lam2
        self.compute_accuracy = compute_accuracy

    def __call__(self, x, t):
        h, t1, t2 = self.calc(x)
        # h: (bs, ch, N), t: (bs, N)
        # print('h', h.shape, 't', t.shape)
        bs, ch, n = h.shape
        h = functions.reshape(functions.transpose(h, (0, 2, 1)), (bs * n, ch))
        t = functions.reshape(t, (bs * n,))
        cls_loss = functions.softmax_cross_entropy(h, t)
        reporter.report({'cls_loss': cls_loss}, self)

        loss = cls_loss
        # Enforce the transformation as orthogonal matrix
        if self.trans and self.trans_lam1 >= 0:
            trans_loss1 = self.trans_lam1 * calc_trans_loss(t1)
            reporter.report({'trans_loss1': trans_loss1}, self)
            loss = loss + trans_loss1
        if self.trans and self.trans_lam2 >= 0:
            trans_loss2 = self.trans_lam2 * calc_trans_loss(t2)
            reporter.report({'trans_loss2': trans_loss2}, self)
            loss = loss + trans_loss2
        reporter.report({'loss': loss}, self)

        if self.compute_accuracy:
            acc = functions.accuracy(h, t)
            reporter.report({'accuracy': acc}, self)
        return loss

    def calc(self, x):
        # x: (minibatch, K, N, 1)
        # N - num_point
        # K - feature degree (this is 3 for xyz input, 64 for middle layer)
        assert x.ndim == 4
        assert x.shape[-1] == 1

        # --- input transform ---
        if self.trans:
            h, t1 = self.input_transform_net(x)
        else:
            h = x
            t1 = 0  # dummy

        h = self.conv_block1(h)
        h = self.conv_block2(h)

        # --- feature transform ---
        if self.trans:
            point_feat, t2 = self.feature_transform_net(h)
        else:
            point_feat = h
            t2 = 0  # dummy

        h = self.conv_block3(point_feat)
        h = self.conv_block4(h)
        h = self.conv_block5(h)

        # Symmetric function: max pooling
        bs, k, n, tmp = h.shape
        assert tmp == 1
        h = functions.max_pooling_2d(h, ksize=h.shape[2:])
        # h: (minibatch, K, 1, 1)
        global_feat = functions.broadcast_to(h, (bs, k, n, 1))
        h = functions.concat([point_feat, global_feat], axis=1)

        h = self.conv_block6(h)
        h = self.conv_block7(h)
        h = self.conv_block8(h)
        h = self.conv_block9(h)
        h = self.conv10(h)
        return h[:, :, :, 0], t1, t2
