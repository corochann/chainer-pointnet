import chainer
from chainer import functions
from chainer import links
from chainer import reporter

from chainer_pointnet.models.conv_block import ConvBlock
from chainer_pointnet.models.pointnet2.feature_propagation_block import \
    FeaturePropagationModule
from chainer_pointnet.models.pointnet2.set_abstraction_block import \
    SetAbstractionModule


class PointNet2SegSSG(chainer.Chain):

    """Segmentation PointNet++ SSG

    Input is (minibatch, K, N, 1), output is (minibatch, out_dim, N)

    Args:
        out_dim (int): output dimension, number of class for classification
        in_dim (int): input dimension for each point. default is 3, (x, y, z).
        dropout_ratio (float): dropout ratio
        use_bn (bool): use batch normalization or not.
        compute_accuracy (bool): compute & report accuracy or not
        residual (bool): use residual connection or not
    """

    def __init__(self, out_dim, in_dim=3, dropout_ratio=0.5,
                 use_bn=True, compute_accuracy=True, residual=False):
        super(PointNet2SegSSG, self).__init__()
        with self.init_scope():
            self.sam1 = SetAbstractionModule(
                k=1024, num_sample_in_region=32, radius=0.1, mlp=[32, 32, 64],
                mlp2=None, use_bn=use_bn, return_distance=True,
                residual=residual)
            self.sam2 = SetAbstractionModule(
                k=256, num_sample_in_region=32, radius=0.2, mlp=[64, 64, 128],
                mlp2=None, use_bn=use_bn, return_distance=True,
                residual=residual)
            self.sam3 = SetAbstractionModule(
                k=64, num_sample_in_region=32, radius=0.4, mlp=[128, 128, 256],
                mlp2=None, use_bn=use_bn, return_distance=True,
                residual=residual)
            self.sam4 = SetAbstractionModule(
                k=16, num_sample_in_region=32, radius=0.8, mlp=[256, 256, 512],
                mlp2=None, use_bn=use_bn, return_distance=True,
                residual=residual)

            self.fpm5 = FeaturePropagationModule(
                mlp=[256, 256], use_bn=use_bn, residual=residual)
            self.fpm6 = FeaturePropagationModule(
                mlp=[256, 256], use_bn=use_bn, residual=residual)
            self.fpm7 = FeaturePropagationModule(
                mlp=[256, 128], use_bn=use_bn, residual=residual)
            self.fpm8 = FeaturePropagationModule(
                mlp=[128, 128, 128], use_bn=use_bn, residual=residual)
            self.conv_block9 = ConvBlock(
                128, 128, ksize=1, use_bn=use_bn, residual=residual)
            self.conv10 = links.Convolution2D(128, out_dim, ksize=1)

        self.compute_accuracy = compute_accuracy

    def calc(self, x):
        # x: (minibatch, K, N, 1)
        # N - num_point
        # K - feature degree (this is 3 for xyz input, 64 for middle layer)
        assert x.ndim == 4
        assert x.shape[-1] == 1

        # TODO: consider support using only XYZ information like
        # coord_points = functions.transpose(x[:, :3, :, 0], (0, 2, 1))
        coord_points = functions.transpose(x[:, :, :, 0], (0, 2, 1))
        # h: feature_points (bs, num_point, ch)
        h0 = None
        coord_points, h1, d1 = self.sam1(coord_points, h0)
        coord_points, h2, d2 = self.sam2(coord_points, h1)
        coord_points, h3, d3 = self.sam3(coord_points, h2)
        coord_points, h4, d4 = self.sam4(coord_points, h3)

        del coord_points
        h3 = self.fpm5(d4, h3, h4)
        del h4, d4
        h2 = self.fpm6(d3, h2, h3)
        del h3, d3
        h1 = self.fpm7(d2, h1, h2)
        del h2, d2
        h0 = self.fpm8(d1, h0, h1)
        del h1, d1
        h = functions.transpose(h0, (0, 2, 1))[:, :, :, None]
        h = self.conv_block9(h)
        h = self.conv10(h)
        return h[:, :, :, 0]

    def __call__(self, x, t):
        h = self.calc(x)

        bs, ch, n = h.shape
        h = functions.reshape(functions.transpose(h, (0, 2, 1)), (bs * n, ch))
        t = functions.reshape(t, (bs * n,))
        cls_loss = functions.softmax_cross_entropy(h, t)
        # reporter.report({'cls_loss': cls_loss}, self)
        loss = cls_loss
        reporter.report({'loss': loss}, self)
        if self.compute_accuracy:
            acc = functions.accuracy(h, t)
            reporter.report({'accuracy': acc}, self)
        return loss

