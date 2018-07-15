import chainer
import numpy
from chainer import functions
from chainer import links
from chainer import reporter

from chainer_pointnet.models.kdcontextnet.kdcontextconv_block import \
    KDContextConvBlock
from chainer_pointnet.models.linear_block import LinearBlock


class KDContextNetCls(chainer.Chain):

    def __init__(self, out_dim, in_dim=3, dropout_ratio=0.5,
                 use_bn=True, compute_accuracy=True):
        super(KDContextNetCls, self).__init__()
        levels = [5, 6, 7]
        levels_diff = numpy.diff(numpy.array([0] + levels))
        feature_learning_mlp_list = [
            [64, 64, 128, 128], [64, 64, 256, 256], [64, 64, 512, 512]]
        feature_aggregation_mlp_list = [[256], [512], [1024]]
        in_channels_list = [in_dim] + [elem[-1] for elem in feature_aggregation_mlp_list]
        fc_mlp_list = [256, 256]
        fcmlps = [in_channels_list[-1]] + fc_mlp_list
        assert len(levels) == len(feature_learning_mlp_list)
        assert len(levels) == len(feature_aggregation_mlp_list)
        with self.init_scope():
            # don't use dropout in conv blocks
            self.kdcontextconv_blocks = chainer.ChainList(
                *[KDContextConvBlock(
                    in_channels_list[i], m=2 ** levels_diff[i],
                    feature_learning_mlp=feature_learning_mlp_list[i],
                    feature_aggregation_mlp=feature_aggregation_mlp_list[i]
                ) for i in range(len(levels_diff))])
            self.fc_blocks = chainer.ChainList(
                *[LinearBlock(
                    fcmlps[i], fcmlps[i+1], use_bn=use_bn,
                    dropout_ratio=dropout_ratio
                ) for i in range(len(fcmlps)-1)])
            self.linear = links.Linear(fcmlps[-1], out_dim)
        self.compute_accuracy = compute_accuracy

    def calc(self, x):
        # x (bs, ch, N, 1)
        assert x.ndim == 4
        assert x.shape[3] == 1
        h = x
        for kdconv_block in self.kdcontextconv_blocks:
            h = kdconv_block(h)
        # h (bs, ch, N//2**levels[-1], 1)
        # TODO: support other symmetric function
        h = functions.max_pooling_2d(h, ksize=(h.shape[2], 1))
        for fc_block in self.fc_blocks:
            h = fc_block(h)
        h = self.linear(h)
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


if __name__ == '__main__':
    import numpy
    from chainer_pointnet.utils.kdtree import construct_kdtree_data

    batchsize = 1
    num_point = 135  # try 100, 128, 135
    max_level = 8  # 2^7 -> 128. Final num_point will be 128
    dim = 3
    point_set = numpy.random.rand(num_point, dim).astype(numpy.float32)
    print('point_set', point_set.shape)
    points, split_dims, inds, kdtree, split_positions = construct_kdtree_data(
        point_set, max_level=max_level, calc_split_positions=True)
    print('points', points.shape)  # 128 point here!
    kdnet = KDContextNetCls(out_dim=5, use_bn=False)
    # split_dims = numpy.array(split_dims)
    # print('split_dims', split_dims.shape, split_dims.dtype)
    pts = numpy.transpose(points, (1, 0))[None, :, :, None]
    print('pts', pts.shape, split_dims.shape)
    out = kdnet.calc(pts)
    print('out', out.shape)




