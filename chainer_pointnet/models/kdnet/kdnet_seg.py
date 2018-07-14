import chainer
from chainer import links, reporter, functions

from chainer_pointnet.models.conv_block import ConvBlock
from chainer_pointnet.models.kdnet.kdconv import KDConv
from chainer_pointnet.models.kdnet.kddeconv import KDDeconv


class KDNetSeg(chainer.Chain):

    """Segmentation KD-Network

    Input is points (minibatch, K, N, 1) and split_dims (minibatch, max_level)
    output is (minibatch, out_dim).
    Here, max_level=log2(N) denotes the level of KDTree.
    `split_dims[i]` is numpy array, represents i-th level split dimension axis.

        Args:
            out_dim (int): output dimension, number of class for classification
            in_dim: input dimension for each point. default is 3, (x, y, z).
            dropout_ratio (float): dropout ratio
            use_bn (bool): use batch normalization or not.
            compute_accuracy (bool): compute & report accuracy or not
            cdim (int): coordinate dimension in KDTree, usually 3 (x, y, z).
    """

    def __init__(self, out_dim, in_dim=3, max_level=10, dropout_ratio=0.0,
                 use_bn=True, compute_accuracy=True, cdim=3):
        super(KDNetSeg, self).__init__()
        if max_level <= 10:
            # depth 10
            ch_list = [in_dim] + [32, 64, 64, 128, 128, 256, 256,
                                  512, 512, 128]
        elif max_level <= 15:
            # depth 15
            ch_list = [in_dim] + [16, 16, 32, 32, 64, 64, 128, 128, 256, 256,
                                  512, 512, 1024, 1024, 128]
        else:
            raise NotImplementedError('depth {} is not implemented yet'
                                      .format(max_level))
        ch_list = ch_list[:max_level + 1]
        out_ch_list = ch_list.copy()
        out_ch_list[0] = 16
        with self.init_scope():
            self.kdconvs = chainer.ChainList(
                *[KDConv(ch_list[i], ch_list[i+1], use_bn=use_bn, cdim=cdim)
                  for i in range(len(ch_list)-1)])
            self.kddeconvs = chainer.ChainList(
                *[KDDeconv(out_ch_list[-i], in_channels_skip=ch_list[-i-1],
                           out_channels=out_ch_list[-i-1], use_bn=use_bn,
                           cdim=cdim)
                  for i in range(1, len(out_ch_list))])
            self.conv_block = ConvBlock(
                out_ch_list[0], out_ch_list[0], ksize=1, use_bn=use_bn)
            self.conv = links.Convolution2D(out_ch_list[0], out_dim, ksize=1)
        self.compute_accuracy = compute_accuracy
        self.max_level = max_level
        self.dropout_ratio = dropout_ratio

    def calc(self, x, split_dims):
        bs = len(split_dims)
        # construct split_dim_list
        split_dim_list = []
        for level in range(self.max_level):
            split_dim_list.append(self.xp.array(
                [split_dims[i, level] for i in range(bs)]))

        h = x
        h_list = [h]
        for d, kdconv in enumerate(self.kdconvs):
            level = self.max_level - d - 1
            h = kdconv(h, split_dim_list[level])
            h_list.append(h)

        h_list.pop(-1)  # don't use last h as skip connection.
        for d, kddeconv in enumerate(self.kddeconvs):
            level = d
            h_skip = h_list.pop(-1)
            # print('h h_skip', h.shape, h_skip.shape, level, len(h_list))
            h = kddeconv(h, split_dim_list[level], h_skip)
        assert len(h_list) == 0

        if self.dropout_ratio > 0.:
            h = functions.dropout(h, self.dropout_ratio)
        h = self.conv_block(h)
        h = self.conv(h)
        return h[:, :, :, 0]

    def __call__(self, x, split_dims, t):
        h = self.calc(x, split_dims)
        bs, ch, n = h.shape
        h = functions.reshape(functions.transpose(h, (0, 2, 1)), (bs * n, ch))
        t = functions.reshape(t, (bs * n,))
        cls_loss = functions.softmax_cross_entropy(h, t)
        loss = cls_loss
        reporter.report({'loss': loss}, self)
        if self.compute_accuracy:
            acc = functions.accuracy(h, t)
            reporter.report({'accuracy': acc}, self)
        return loss


if __name__ == '__main__':
    import numpy
    from chainer_pointnet.utils.kdtree import construct_kdtree_data

    batchsize = 1
    num_point = 135  # try 100, 128, 135
    max_level = 7  # 2^7 -> 128. Final num_point will be 128
    dim = 3
    point_set = numpy.random.rand(num_point, dim).astype(numpy.float32)
    print('point_set', point_set.shape)
    points, split_dims, inds, kdtree, split_positions = construct_kdtree_data(
        point_set, max_level=max_level, calc_split_positions=True)
    print('points', points.shape)  # 128 point here!
    kdnet = KDNetSeg(3, max_level=max_level, use_bn=False)
    split_dims = numpy.array(split_dims)
    print('split_dims', split_dims.shape, split_dims.dtype)
    pts = numpy.transpose(points, (1, 0))[None, :, :, None]
    print('pts', pts.shape, split_dims.shape)
    out = kdnet.calc(pts, split_dims[None, ...])
    print('out', out.shape)


