import chainer
from chainer import functions, links


class KDConv(chainer.Chain):

    """KD-Convolution. apply eq (1) or (2) of the paper

    Escape from Cells: Deep Kd-Networks for the Recognition of
    3D Point Cloud Models.
    """

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, use_bn=True,
                 activation=functions.relu, cdim=3):
        # cdim: coordinate dimension, usually 3 (x, y, z).
        super(KDConv, self).__init__()
        in_ch = None if in_channels is None else in_channels * 2
        with self.init_scope():
            self.conv = links.Convolution2D(
                in_ch, out_channels * cdim, ksize=ksize,
                stride=stride, pad=pad, nobias=nobias, initialW=initialW,
                initial_bias=initial_bias)
            if use_bn:
                self.bn = links.BatchNormalization(out_channels)
        self.out_channels = out_channels
        self.cdim = cdim
        self.activation = activation
        self.use_bn = use_bn

    def __call__(self, x, split_dim):
        """KDConv makes feature of 1-level parent node of KDTree.

        `num_point` of input `x` will be half in the `output`.

        Args:
            x (numpy.ndarray):
                4-dim array (batchsize, in_channels, num_point, 1)
            split_dim (numpy.ndarray): 1d array with dtype=object with
                length=max_level. `split_dim[i]` is numpy array with dtype=int,
                represents i-th level split dimension axis.

        Returns (numpy.ndarray):
            4-dim array (batchsize, out_channels, num_point//2, 1)

        """
        assert x.ndim == 4
        assert x.shape[2] // 2 == split_dim.shape[1],\
            'x.shape {}, split_dim.shape{}'.format(x.shape, split_dim.shape)
        # x: (batch_size, ch, num_point, 1)
        bs, ch, num_point, w = x.shape
        assert w == 1
        x = functions.reshape(x, (bs, ch, num_point//2, 2))
        x = functions.transpose(x, (0, 1, 3, 2))
        x = functions.reshape(x, (bs, ch * 2, num_point//2, 1))
        x = self.conv(x)
        # split_dim: (batch_size, num_point//2) dtype=np.int32
        # x: (batch_size, out_channels, cdim, num_point//2, 1)
        x = functions.reshape(x, (bs, self.out_channels, self.cdim,
                                  num_point//2))
        # select `split_dim`'s output (extract KDTree's split axis conv result)
        x = x[self.xp.arange(bs)[:, None], :, split_dim,
              self.xp.arange(num_point//2)[None, :]]
        # x: (batch_size, num_point//2, out_channels)
        x = functions.transpose(x, (0, 2, 1))
        x = functions.reshape(x, (bs, self.out_channels, num_point//2, 1))
        # x: (batch_size, out_channels, num_point//2, 1)
        if self.use_bn:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


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
    kdconv = KDConv(3, 8)
    split_dim = numpy.array([split_dims[-1], split_dims2[-1]])
    print('split_dim', split_dim.shape)
    pts = numpy.array([points, points2])
    pts = numpy.transpose(pts, (0, 2, 1))[:, :, :, None]
    print('pts', pts.shape)
    out = kdconv(pts, split_dim)
    print('out', out.shape)
