import chainer
from chainer import functions
from chainer import links
from chainer import reporter

from chainer_pointnet.models.linear_block import LinearBlock
from chainer_pointnet.models.pointnet2.set_abstraction_all_block import \
    SetAbstractionGroupAllModule
from chainer_pointnet.models.pointnet2.set_abstraction_block import \
    SetAbstractionModule


class PointNet2ClsSSG(chainer.Chain):

    """Classification PointNet++ SSG

    Input is (minibatch, K, N, 1), output is (minibatch, out_dim)

    Args:
        out_dim (int): output dimension, number of class for classification
        in_dim: input dimension for each point. default is 3, (x, y, z).
        dropout_ratio (float): dropout ratio
        use_bn (bool): use batch normalization or not.
        compute_accuracy (bool): compute & report accuracy or not
        residual (bool): use residual connection or not
    """

    def __init__(self, out_dim, in_dim=3, dropout_ratio=0.5,
                 use_bn=True, compute_accuracy=True, residual=False):
        super(PointNet2ClsSSG, self).__init__()
        with self.init_scope():
            self.sam1 = SetAbstractionModule(
                k=512, num_sample_in_region=32, radius=0.2,
                mlp=[64, 64, 128], mlp2=None, residual=residual)
            self.sam2 = SetAbstractionModule(
                k=128, num_sample_in_region=64, radius=0.4,
                mlp=[128, 128, 256], mlp2=None, residual=residual)
            # k, num_sample_in_region, radius are ignored when group_all=True
            self.sam3 = SetAbstractionGroupAllModule(
                mlp=[256, 512, 1024], mlp2=None, residual=residual)

            self.fc_block4 = LinearBlock(
                1024, 512, use_bn=use_bn, dropout_ratio=dropout_ratio,)
            self.fc_block5 = LinearBlock(
                512, 256, use_bn=use_bn, dropout_ratio=dropout_ratio,)
            self.fc6 = links.Linear(256, out_dim)

        self.compute_accuracy = compute_accuracy

    def calc(self, x):
        # x: (minibatch, K, N, 1)
        # N - num_point
        # K - feature degree (this is 3 for xyz input, 64 for middle layer)
        assert x.ndim == 4
        assert x.shape[-1] == 1

        coord_points = functions.transpose(x[:, :, :, 0], (0, 2, 1))
        # h: feature_points
        h = None
        coord_points, h, _ = self.sam1(coord_points, h)
        coord_points, h, _ = self.sam2(coord_points, h)
        coord_points, h = self.sam3(coord_points, h)
        # coord (bs, k, coord), h: feature (bs, k, ch')
        h = self.fc_block4(h)
        h = self.fc_block5(h)
        h = self.fc6(h)
        return h

    def __call__(self, x, t):
        h = self.calc(x)
        cls_loss = functions.softmax_cross_entropy(h, t)
        # reporter.report({'cls_loss': cls_loss}, self)
        loss = cls_loss
        reporter.report({'loss': loss}, self)
        if self.compute_accuracy:
            acc = functions.accuracy(h, t)
            reporter.report({'accuracy': acc}, self)
        return loss

