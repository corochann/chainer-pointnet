import chainer
from chainer import functions

from chainer_pointnet.models.conv_block import ConvBlock


class KDContextConvBlock(chainer.Chain):

    """KD Context Convolution Block

    It performs each level feature learning & aggregation

    Args:
        in_channels (int or None): input channels
        m (int): Number to group to be aggregated.
        feature_learning_mlp (list): list of int, specifies MLP size for
            feature learning stage.
        feature_aggregation_mlp (list): list of int, specifies MLP size for
            feature aggregation stage.
        ksize (int or tuple): kernel size
        stride (int or tuple): stride size
        pad (int or tuple): padding size
        nobias (bool): use bias `b` or not.
        initialW: initiallizer of `W`
        initial_bias: initializer of `b`
        use_bn (bool): use batch normalization or not
        activation (callable): activation function
        dropout_ratio (float): dropout ratio, set negative value to skip
            dropout
        aggregation (bool): apply aggregation or not.
            Default is `True`, `num_point` will be `num_point//m` in output.
            `False` is set for deconvolution part, not reduce number of points.
    """

    def __init__(self, in_channels, m,
                 feature_learning_mlp=None,
                 feature_aggregation_mlp=None,
                 ksize=1, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, use_bn=True,
                 activation=functions.relu, dropout_ratio=-1,
                 aggregation=True):
        super(KDContextConvBlock, self).__init__()
        if feature_learning_mlp is None:
            print('feature_learning_mlp is None, value set automatically')
            feature_learning_mlp = [16]
        if feature_aggregation_mlp is None:
            print('feature_aggregation_mlp is None, value set automatically')
            feature_aggregation_mlp = [256]

        # must be multiple of 4.
        # Used for (y_i, \sigma(g(Y)), G(x_i)-\theta&\phi, H(x_i)) respectively
        # First 2 are for Local Contextual Cues and
        # Latter 2 are for Global Contextual Cues.
        assert feature_learning_mlp[-1] % 4 == 0
        # TODO: support dense-net calculation for flmlp
        flmlp = [in_channels] + feature_learning_mlp
        famlp = [flmlp[-1]//2] + feature_aggregation_mlp
        with self.init_scope():
            self.flconvs = chainer.ChainList(
                *[ConvBlock(flmlp[i], flmlp[i+1], ksize=ksize, stride=stride,
                            pad=pad, nobias=nobias, initialW=initialW,
                            initial_bias=initial_bias, use_bn=use_bn,
                            activation=activation, dropout_ratio=dropout_ratio,
                            ) for i in range(len(flmlp)-1)])
            self.faconvs = chainer.ChainList(
                *[ConvBlock(famlp[i], famlp[i + 1], ksize=ksize, stride=stride,
                            pad=pad, nobias=nobias, initialW=initialW,
                            initial_bias=initial_bias, use_bn=use_bn,
                            activation=activation, dropout_ratio=dropout_ratio,
                            ) for i in range(len(famlp)-1)])
        self.m = m
        self.aggregation = aggregation

    def __call__(self, x):
        """Main forward computation

        Args:
            x (array or Variable): 4-dim array (minibatch, K, N, 1).
                `K=in_channels` is channel, `N` is number of points.

        Returns (array or Variable): 4-dim array (minibatch, K', N//m, 1).
            `K'=feature_aggregation_mlp[-1]` is channel,
            `N` is number of input points.
            Number of output points are reduced to `N//m` after feature
            aggregation.
            If `aggregation=False`, then output shape is (minibatch, K', N, 1)

        """
        assert x.ndim == 4  # (bs, ch, N, 1)
        assert x.shape[3] == 1
        h = x
        # Feature Learning Stage
        for conv in self.flconvs:
            h = conv(h)
        h0, h1, h2, h3 = functions.split_axis(h, 4, axis=1)
        # 1. Local Contextual Cues
        # TODO: support other symmetric function
        gy = functions.max_pooling_2d(h0, ksize=(self.m, 1))
        bs, ch, n, _ = h1.shape
        assert n % self.m == 0
        h1 = functions.reshape(h1, (bs, ch, n//self.m, self.m))
        h_local = functions.broadcast_to(functions.sigmoid(gy), h1.shape) * h1
        del gy, h0, h1
        h_local = functions.reshape(h_local, (bs, ch, n, 1))
        # 2. Global Contextual Cues
        # See also https://arxiv.org/pdf/1711.07971.pdf
        # h2, h3 (batchsize, ch, N, 1) -> (bs, ch, N, m)
        # h2 is used for both \theta(x_i) and \phi(x_j).
        # h3 is used for H(x_j) of Eq(3) of 3DContextNet paper.
        bs, ch, n, _ = h2.shape
        assert n % self.m == 0

        # TODO: support other symmetric function
        # h2, h3 (batchsize, ch, N, 1) -> (bs, ch, N//m)
        h2 = functions.max_pooling_2d(h2, ksize=(self.m, 1))[:, :, :, 0]
        h3 = functions.max_pooling_2d(h3, ksize=(self.m, 1))[:, :, :, 0]

        # g (bs, hw, hw), where hw=`n//self.m`
        g = functions.matmul(h2, h2, transa=True)
        # g(x_i, x_j) / C(x) can be replaced by softmax calculation.
        h_global = functions.matmul(h3, functions.softmax(g, axis=1))
        del g, h2, h3
        # (bs, ch, hw=N//m) -> (bs, ch, N, 1)
        h_global = functions.reshape(h_global, (bs, ch, n//self.m, 1))
        h_global = functions.broadcast_to(h_global, (bs, ch, n//self.m, self.m))
        h_global = functions.reshape(h_global, (bs, ch, n, 1))

        # Feature Aggregation Stage
        h = functions.concat([h_local, h_global], axis=1)
        del h_local, h_global
        for conv in self.faconvs:
            h = conv(h)
        if self.aggregation:
            # TODO: support other symmetric function
            h = functions.max_pooling_2d(h, ksize=(self.m, 1))
        return h


if __name__ == '__main__':
    import numpy
    from chainer_pointnet.utils.kdtree import construct_kdtree_data

    batchsize = 1
    num_point = 135  # try 100, 128, 135
    max_level = 7  # 2^7 -> 128. Final num_point will be 128
    dim = 3
    point_set = numpy.random.rand(num_point, dim).astype(numpy.float32)
    point_set2 = numpy.random.rand(num_point, dim).astype(numpy.float32)
    print('point_set', point_set.shape)
    points, split_dims, inds, kdtree, split_positions = construct_kdtree_data(
        point_set, max_level=7, calc_split_positions=True)
    points2, split_dims2, inds2, kdtree2, split_positions2 = construct_kdtree_data(
        point_set2, max_level=7, calc_split_positions=True)
    print('points', points.shape)  # 128 point here!
    kdconvblock = KDContextConvBlock(
        3, m=2**3, feature_learning_mlp=[16], feature_aggregation_mlp=[24])
    # split_dim = numpy.array([split_dims[-1], split_dims2[-1]])
    # print('split_dim', split_dim.shape)
    pts = numpy.array([points, points2])
    pts = numpy.transpose(pts, (0, 2, 1))[:, :, :, None]
    print('pts', pts.shape)
    out = kdconvblock(pts)
    print('out', out.shape)
