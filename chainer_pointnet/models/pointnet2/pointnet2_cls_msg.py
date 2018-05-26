import numpy

import chainer
from chainer import functions, cuda
from chainer import links
from chainer import reporter

from chainer_pointnet.models.linear_block import LinearBlock
from chainer_pointnet.models.pointnet2.set_abstraction_all_block import \
    SetAbstractionGroupAllModule
from chainer_pointnet.models.pointnet2.set_abstraction_block import \
    SetAbstractionModule


class PointNet2ClsMSG(chainer.Chain):

    """Classification PointNet++ MSG (Multi Scale Grouping)

    Input is (minibatch, K, N, 1), output is (minibatch, out_dim)

        Args:
            out_dim (int): output dimension, number of class for classification
            in_dim: input dimension for each point. default is 3, (x, y, z).
            dropout_ratio (float): dropout ratio
            compute_accuracy (bool): compute & report accuracy or not
    """

    def __init__(self, out_dim, in_dim=3, dropout_ratio=0.5,
                 use_bn=True, compute_accuracy=True):
        super(PointNet2ClsMSG, self).__init__()
        with self.init_scope():
            # initial_idx is set to ensure deterministic behavior of
            # fathest_point_sampling
            self.sam11 = SetAbstractionModule(
                k=512, num_sample_in_region=16, radius=0.1,
                mlp=[32, 32, 64], mlp2=None, initial_idx=0)
            self.sam12 = SetAbstractionModule(
                k=512, num_sample_in_region=32, radius=0.2,
                mlp=[64, 64, 128], mlp2=None, initial_idx=0)
            self.sam13 = SetAbstractionModule(
                k=512, num_sample_in_region=128, radius=0.4,
                mlp=[64, 96, 128], mlp2=None, initial_idx=0)
            self.sam21 = SetAbstractionModule(
                k=128, num_sample_in_region=32, radius=0.2,
                mlp=[64, 64, 128], mlp2=None, initial_idx=0)
            self.sam22 = SetAbstractionModule(
                k=128, num_sample_in_region=64, radius=0.4,
                mlp=[128, 128, 256], mlp2=None, initial_idx=0)
            self.sam23 = SetAbstractionModule(
                k=128, num_sample_in_region=128, radius=0.8,
                mlp=[128, 128, 256], mlp2=None, initial_idx=0)
            self.sam3 = SetAbstractionGroupAllModule(
                mlp=[256, 512, 1024], mlp2=None)

            self.fc_block4 = LinearBlock(
                1024, 512, use_bn=use_bn, dropout_ratio=dropout_ratio)
            self.fc_block5 = LinearBlock(
                512, 256, use_bn=use_bn, dropout_ratio=dropout_ratio)
            self.fc6 = links.Linear(256, out_dim)

        self.compute_accuracy = compute_accuracy

    def calc(self, x):
        # x: (minibatch, K, N, 1)
        # N - num_point
        # K - feature degree (this is 3 for xyz input, 64 for middle layer)
        assert x.ndim == 4
        assert x.shape[-1] == 1

        coord_points = functions.transpose(x[:, :, :, 0], (0, 2, 1))
        feature_points = None
        cp11, fp11, _ = self.sam11(coord_points, feature_points)
        cp12, fp12, _ = self.sam12(coord_points, feature_points)
        cp13, fp13, _ = self.sam13(coord_points, feature_points)
        # assert numpy.allclose(cuda.to_cpu(cp11.data), cuda.to_cpu(cp12.data))
        # assert numpy.allclose(cuda.to_cpu(cp11.data), cuda.to_cpu(cp13.data))
        del cp12, cp13

        feature_points = functions.concat([fp11, fp12, fp13], axis=2)
        cp21, fp21, _ = self.sam21(cp11, feature_points)
        cp22, fp22, _ = self.sam21(cp11, feature_points)
        cp23, fp23, _ = self.sam21(cp11, feature_points)
        # assert numpy.allclose(cuda.to_cpu(cp21.data), cuda.to_cpu(cp22.data))
        # assert numpy.allclose(cuda.to_cpu(cp21.data), cuda.to_cpu(cp23.data))
        del cp22, cp23
        feature_points = functions.concat([fp21, fp22, fp23], axis=2)

        coord_points, feature_points = self.sam3(cp21, feature_points)
        # coord (bs, k, coord), feature (bs, k, ch')
        h = self.fc_block4(feature_points)
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

