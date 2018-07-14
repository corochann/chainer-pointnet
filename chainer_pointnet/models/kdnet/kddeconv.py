import chainer
from chainer import functions, links


class KDDeconv(chainer.Chain):

    """KD-Deconvolution. apply eq (4) of the paper

    Escape from Cells: Deep Kd-Networks for the Recognition of
    3D Point Cloud Models.
    """

    def __init__(self, in_channels, out_channels, in_channels_skip=None,
                 ksize=1, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, use_bn=True,
                 activation=functions.relu, cdim=3):
        # cdim: coordinate dimension, usually 3 (x, y, z).
        super(KDDeconv, self).__init__()
        out_ch = out_channels // 2
        out_channels_skip = out_channels - out_ch
        with self.init_scope():
            self.conv = links.Convolution2D(
                in_channels, out_ch * cdim * 2, ksize=ksize,
                stride=stride, pad=pad, nobias=nobias, initialW=initialW,
                initial_bias=initial_bias)
            self.conv_skip = links.Convolution2D(
                in_channels_skip, out_channels_skip, ksize=ksize,
                stride=stride, pad=pad, nobias=nobias, initialW=initialW,
                initial_bias=initial_bias)
            if use_bn:
                self.bn = links.BatchNormalization(out_channels)
        self.out_ch = out_ch
        self.out_channels = out_channels
        self.cdim = cdim
        self.activation = activation
        self.use_bn = use_bn

    def __call__(self, x, split_dim, x_skip):
        """KDDeconv makes feature of 1-level children node of KDTree.

        `num_point` of input `x` will be twice in the `output`.

        Args:
            x (numpy.ndarray):
                4-dim array (batchsize, in_channels, num_point, 1)
            split_dim (numpy.ndarray): 1d array with dtype=object with
                length=max_level. `split_dim[i]` is numpy array with dtype=int,
                represents i-th level split dimension axis.
            x_skip (numpy.ndarray):
                4-dim array (batchsize, in_channels_skip, num_point*2, 1)

        Returns (numpy.ndarray):
            4-dim array (batchsize, out_channels, num_point*2, 1)

        """
        assert x.ndim == 4
        assert x_skip.ndim == 4
        assert x.shape[2] == split_dim.shape[1],\
            'x.shape {}, split_dim.shape{}'.format(x.shape, split_dim.shape)
        # x: (batch_size, ch, num_point, 1)
        bs, ch, num_point, w = x.shape
        bs, ch_skip, num_point2, w_skip = x_skip.shape
        assert w == 1
        assert w_skip == 1
        assert num_point * 2 == num_point2

        # --- deconvolution from parent node to children part ---
        # (bs, ch, N, 1) -> (bs, out_ch, N*2, 1)

        # conv: (bs, ch, N, 1) -> (bs, out_ch*cdim*2, N, 1)
        h = self.conv(x)
        # select split_dim:    -> (bs, out_ch*2, N, 1)
        h = functions.reshape(h, (bs, self.out_ch, self.cdim, 2, num_point))
        h = h[self.xp.arange(bs)[:, None], :, split_dim, :,
              self.xp.arange(num_point)[None, :]]
        # h (bs, num_point, out_ch, 2)
        h = functions.transpose(h, (0, 2, 1, 3))
        h = functions.reshape(h, (bs, self.out_ch, num_point2, 1))
        # split to child node: -> (bs, out_ch,   N*2, 1)

        # --- skip connection part ---
        # (bs, ch_skip, N*2, 1) -> (bs, out_channels_skip, N*2, 1)
        h_skip = self.conv_skip(x_skip)
        # concat deconv feature `h` and skip connection feature `h_skip`
        h = functions.concat([h, h_skip], axis=1)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


if __name__ == '__main__':
    import numpy
    from chainer_pointnet.utils.kdtree import construct_kdtree_data
    from chainer_pointnet.models.kdnet.kdconv import KDConv

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
    # --- net definition ---
    kdconv = KDConv(3, 8)
    kddeconv = KDDeconv(8, out_channels=11, in_channels_skip=3)
    # --- net definition end ---

    split_dim = numpy.array([split_dims[-1], split_dims2[-1]])
    print('split_dim', split_dim.shape)
    pts = numpy.array([points, points2])
    pts = numpy.transpose(pts, (0, 2, 1))[:, :, :, None]
    print('pts', pts.shape)
    pts2 = kdconv(pts, split_dim)
    print('pts2', pts2.shape)
    out = kddeconv(pts2, split_dim, pts)
    print('out', out.shape)

