import chainer
from chainer import links, reporter, functions

from chainer_pointnet.models.kdnet.kdconv import KDConv


class KDNetCls(chainer.Chain):

    """Classification KD-Network

    Input is (minibatch, K, N, 1), output is (minibatch, out_dim)

        Args:
            out_dim (int): output dimension, number of class for classification
            in_dim: input dimension for each point. default is 3, (x, y, z).
            dropout_ratio (float): dropout ratio
            compute_accuracy (bool): compute & report accuracy or not
    """

    def __init__(self, out_dim, in_dim=3, dropout_ratio=0.5,
                 compute_accuracy=True, depth=10):
        super(KDNetCls, self).__init__()
        if depth <= 10:
            # depth 10
            ch_list = [in_dim] + [32, 64, 64, 128, 128, 256, 256,
                                  512, 512, 128]
            ch_list = ch_list[:depth+1]
        elif depth <= 15:
            # depth 15
            ch_list = [in_dim] + [16, 16, 32, 32, 64, 64, 128, 128, 256, 256,
                                  512, 512, 1024, 1024, 128]
            ch_list = ch_list[:depth + 1]
        else:
            raise NotImplementedError('depth {} is not implemented yet'
                                      .format(depth))
        with self.init_scope():
            self.kdconvs = chainer.ChainList(
                *[KDConv(ch_list[i], ch_list[i+1]) for i in
                  range(len(ch_list)-1)])
            self.linear = links.Linear(None, out_dim)
        self.compute_accuracy = compute_accuracy
        self.depth = depth

    def calc(self, x, split_dims):
        bs = len(split_dims)
        h = x
        for d, kdconv in enumerate(self.kdconvs):
            level = self.depth - d - 1
            split_dim = self.xp.array(
                [split_dims[i, level] for i in range(bs)])
            h = kdconv(h, split_dim)
        return self.linear(h)

    def __call__(self, x, split_dims, t):
        h = self.calc(x, split_dims)
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
    points, split_dims, kdtree, split_positions = construct_kdtree_data(
        point_set, max_level=7, calc_split_positions=True)
    print('points', points.shape)  # 128 point here!
    # kdconv = KDConv(3, 8)
    kdnet = KDNetCls(3, depth=7)
    split_dims = numpy.array(split_dims)
    print('split_dims', split_dims.shape, split_dims.dtype)
    pts = numpy.transpose(points, (1, 0))[None, :, :, None]
    print('pts', pts.shape, split_dims.shape)
    out = kdnet.calc(pts, split_dims[None, ...])
    print('out', out.shape)

