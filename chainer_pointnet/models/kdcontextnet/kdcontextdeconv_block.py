import chainer
from chainer import functions
from chainer import links

from chainer_pointnet.models.conv_block import ConvBlock
from chainer_pointnet.models.kdcontextnet.kdcontextconv_block import \
    KDContextConvBlock


class KDContextDeconvBlock(chainer.Chain):

    """KD Context Deconvolution Block

    It performs each level feature learning & aggregation

    Args:
        in_channels (int or None): channel size for input `x`
        m (int): Number to group to be aggregated.
        out_deconv_channels (int or None): output channels size for deconv part
        in_channels_skip (int or None): channel size for input `x_skip`
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
    """

    def __init__(self, in_channels, m, in_channels_skip=None,
                 out_deconv_channels=None,
                 feature_learning_mlp=None,
                 feature_aggregation_mlp=None,
                 ksize=1, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, use_bn=True,
                 activation=functions.relu, dropout_ratio=-1):
        super(KDContextDeconvBlock, self).__init__()
        if in_channels_skip is None:
            in_channels_context = None
        else:
            in_channels_context = in_channels_skip + out_deconv_channels
        if out_deconv_channels is None:
            raise ValueError('currently out_deconv_channels must be set '
                             'specifically')
        with self.init_scope():
            # deconvolution part
            self.conv = links.Convolution2D(
                in_channels, out_deconv_channels * m, ksize=ksize,
                stride=stride, pad=pad, nobias=nobias, initialW=initialW,
                initial_bias=initial_bias)
            # set `aggregation=False` to keep num_point of output.
            self.kdcontextconv = KDContextConvBlock(
                in_channels_context, m,
                feature_learning_mlp=feature_learning_mlp,
                feature_aggregation_mlp=feature_aggregation_mlp,
                ksize=ksize, stride=stride, pad=pad,
                nobias=nobias, initialW=initialW, initial_bias=initial_bias,
                use_bn=use_bn, activation=activation,
                dropout_ratio=dropout_ratio, aggregation=False)
        self.m = m
        self.out_deconv_channels = out_deconv_channels

    def __call__(self, x, x_skip):
        """Main forward computation

        Args:
            x (array or Variable): Input from parent node.
                4-dim array (batchsize, in_channels, num_point, 1)
            x_skip (array or Variable): Input from skip connection.
                4-dim array (batchsize, in_channels_skip, num_point*m, 1)

        Returns (array or Variable):
            4-dim array (batchsize, out_channels, num_point*m, 1)
            Number of output points are increased to `num_point*m` after
            deconvolution.
        """
        assert x.ndim == 4  # (bs, ch, N, 1)
        assert x_skip.ndim == 4  # (bs, ch, N, 1)
        bs, ch, num_point, w = x.shape
        bs, ch_skip, num_pointm, w_skip = x_skip.shape
        assert w == 1
        assert w_skip == 1
        assert num_point * self.m == num_pointm
        # --- deconvolution from parent node to children part ---
        # (bs, ch, n, 1) -> (bs, out_ch*m, n, 1) -> (bs, out_ch, n*m, 1)
        h = self.conv(x)
        h = functions.reshape(h, (bs, self.out_deconv_channels,
                                  self.m * num_point, 1))
        # --- skip connection part ---
        # Do nothing, just concat.
        # Other way is to put one convolution operation,
        # and we can reduce channels for `h_skip` here.
        # h_skip = self.conv_skip(x_skip)
        h_skip = x_skip  # (bs, ch_skip, n*m, 1)

        h = functions.concat([h, h_skip], axis=1)
        h = self.kdcontextconv(h)
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
    kddeconvblock = KDContextDeconvBlock(
        24, m=2**3, out_deconv_channels=12,
        feature_learning_mlp=[28], feature_aggregation_mlp=[30])
    # split_dim = numpy.array([split_dims[-1], split_dims2[-1]])
    # print('split_dim', split_dim.shape)
    pts = numpy.array([points, points2])
    pts = numpy.transpose(pts, (0, 2, 1))[:, :, :, None]
    print('pts', pts.shape)  # (2, 3, 128, 1)
    pts2 = kdconvblock(pts)
    print('pts2', pts2.shape)  # (2, 24, 16, 1)
    out = kddeconvblock(pts2, pts)
    print('out', out.shape)  # (2, 30, 128, 1)
