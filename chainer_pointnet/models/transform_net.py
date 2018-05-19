import numpy

import chainer
from chainer import functions
from chainer import initializers
from chainer import links

from chainer_pointnet.models.conv_block import ConvBlock


class TransformModule(chainer.Chain):

    """Transform module

    This class produces transform matrix
    Input is (minibatch, K, N, 1), output is (minibatch, K, K)
    """

    def __init__(self, k=3, use_bn=True):
        super(TransformModule, self).__init__()
        initial_bias = numpy.identity(k, dtype=numpy.float32).ravel()
        with self.init_scope():
            self.conv_block1 = ConvBlock(k, 64, ksize=1, use_bn=use_bn)
            self.conv_block2 = ConvBlock(64, 128, ksize=1, use_bn=use_bn)
            self.conv_block3 = ConvBlock(128, 1024, ksize=1, use_bn=use_bn)
            # [Note]
            # Original paper uses BN for fc layer as well.
            # https://github.com/charlesq34/pointnet/blob/master/models/transform_nets.py#L34
            # This chanier impl. skip BN for fc layer
            self.fc4 = links.Linear(1024, 512)
            # self.bn4 = links.BatchNormalization(512)
            self.fc5 = links.Linear(512, 256)
            # self.bn5 = links.BatchNormalization(256)

            # initial output of transform net should be identity
            self.fc6 = links.Linear(
                256, k * k, initialW=initializers.Zero(dtype=numpy.float32),
                initial_bias=initial_bias)
        self.k = k

    def __call__(self, x):
        # reference --> x: (minibatch, N, 1, K) <- original tf impl.
        # x: (minibatch, K, N, 1) <- chainer impl.
        # N - num_point
        # K - feature degree (this is 3 for xyz input, 64 for middle layer)
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        h = functions.max_pooling_2d(h, ksize=h.shape[2:])
        # h: (minibatch, K, 1, 1)
        h = functions.relu(self.fc4(h))
        h = functions.relu(self.fc5(h))
        h = self.fc6(h)
        bs, k2 = h.shape
        assert k2 == self.k ** 2
        h = functions.reshape(h, (bs, self.k, self.k))
        return h


class TransformNet(chainer.Chain):
    """Transform Network

    This class can be used for Both InputTransformNet & FeatureTransformNet
    Input is (minibatch, K, N, 1),
    output is (minibatch, K, N, 1), which is transformed
    """

    def __init__(self, k=3, use_bn=True):
        super(TransformNet, self).__init__()
        with self.init_scope():
            self.trans_module = TransformModule(k=k, use_bn=use_bn)

    def __call__(self, x):
        t = self.trans_module(x)
        # t: (minibatch, K, K)
        # x: (minibatch, K, N, 1)
        # h: (minibatch, K, N)
        # K = in_dim
        h = functions.matmul(t, x[:, :, :, 0])
        bs, k, n = h.shape
        h = functions.reshape(h, (bs, k, n, 1))
        return h, t
