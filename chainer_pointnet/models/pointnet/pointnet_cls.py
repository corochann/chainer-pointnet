import chainer
from chainer import functions, cuda
from chainer import links
from chainer import reporter

from chainer_pointnet.models.conv_block import ConvBlock
from chainer_pointnet.models.linear_block import LinearBlock
from chainer_pointnet.models.pointnet.transform_net import TransformNet


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


class PointNetCls(chainer.Chain):

    """Classification PointNet

    Input is (minibatch, K, N, 1), output is (minibatch, out_dim)

        Args:
            out_dim (int): output dimension, number of class for classification
            in_dim: input dimension for each point. default is 3, (x, y, z).
            middle_dim (int): hidden layer
            dropout_ratio (float): dropout ratio, negative value indicates
                not to use dropout.
            use_bn (bool): use batch normalization or not.
            trans (bool): use TransformNet or not.
                False means not to use TransformNet, corresponds to
                PointNetVanilla. True corresponds to PointNet in the paper.
            trans_lam1 (float): regularization term for input transform.
                used in training. it is simply ignored when `trans` is False.
            trans_lam2 (float): regularization term for feature transform
                used in training. it is simply ignored when `trans` is False.
            compute_accuracy (bool): compute & report accuracy or not
            residual (bool): use residual connection or not
    """

    def __init__(self, out_dim, in_dim=3, middle_dim=64, dropout_ratio=0.3,
                 use_bn=True, trans=True, trans_lam1=0.001, trans_lam2=0.001,
                 compute_accuracy=True, residual=False):
        super(PointNetCls, self).__init__()
        with self.init_scope():
            if trans:
                self.input_transform_net = TransformNet(
                    k=in_dim, use_bn=use_bn, residual=residual)

            self.conv_block1 = ConvBlock(
                in_dim, 64, ksize=1, use_bn=use_bn, residual=residual)
            self.conv_block2 = ConvBlock(
                64, middle_dim, ksize=1, use_bn=use_bn, residual=residual)
            if trans:
                self.feature_transform_net = TransformNet(
                    k=middle_dim, use_bn=use_bn, residual=residual)

            self.conv_block3 = ConvBlock(
                middle_dim, 64, ksize=1, use_bn=use_bn, residual=residual)
            self.conv_block4 = ConvBlock(
                64, 128, ksize=1, use_bn=use_bn, residual=residual)
            self.conv_block5 = ConvBlock(
                128, 1024, ksize=1, use_bn=use_bn, residual=residual)

            # original impl. uses `keep_prob=0.7`.
            self.fc_block6 = LinearBlock(
                1024, 512, use_bn=use_bn, dropout_ratio=dropout_ratio,
                residual=residual)
            self.fc_block7 = LinearBlock(
                512, 256, use_bn=use_bn, dropout_ratio=dropout_ratio,
                residual=residual)
            self.fc8 = links.Linear(256, out_dim)

        self.in_dim = in_dim
        self.trans = trans
        self.trans_lam1 = trans_lam1
        self.trans_lam2 = trans_lam2
        self.compute_accuracy = compute_accuracy

    def __call__(self, x, t):
        h, t1, t2 = self.calc(x)
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
            h, t2 = self.feature_transform_net(h)
        else:
            t2 = 0  # dummy

        h = self.conv_block3(h)
        h = self.conv_block4(h)
        h = self.conv_block5(h)

        # Symmetric function: max pooling
        h = functions.max_pooling_2d(h, ksize=h.shape[2:])
        # h: (minibatch, K, 1, 1)
        h = self.fc_block6(h)
        h = self.fc_block7(h)
        h = self.fc8(h)
        return h, t1, t2
