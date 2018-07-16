import numpy

import chainer
from chainer import functions
from chainer import links

from chainer_pointnet.models.conv_block import ConvBlock
from chainer_pointnet.utils.grouping import query_ball_by_diff
from chainer_pointnet.utils.sampling import farthest_point_sampling


class SetAbstractionGroupAllModule(chainer.Chain):

    def __init__(self, mlp, mlp2, in_channels=None, use_bn=True,
                 activation=functions.relu, residual=False):
        # k is number of sampled point (num_region)
        super(SetAbstractionGroupAllModule, self).__init__()
        # Feature Extractor channel list
        assert isinstance(mlp, list)
        fe_ch_list = [in_channels] + mlp
        # Head channel list
        if mlp2 is None:
            mlp2 = []
        assert isinstance(mlp2, list)
        head_ch_list = [mlp[-1]] + mlp2
        with self.init_scope():
            self.sampling_grouping = SamplingGroupingAllModule()
            self.feature_extractor_list = chainer.ChainList(
                *[ConvBlock(fe_ch_list[i], fe_ch_list[i+1], ksize=1,
                            use_bn=use_bn, activation=activation,
                            residual=residual
                            ) for i in range(len(mlp))])
            self.head_list = chainer.ChainList(
                *[ConvBlock(head_ch_list[i], head_ch_list[i + 1], ksize=1,
                            use_bn=use_bn, activation=activation,
                            residual=residual
                            ) for i in range(len(mlp2))])
        self.use_bn = use_bn

    def __call__(self, coord_points, feature_points=None):
        # coord_points   (batch_size, num_point, coord_dim)
        # feature_points (batch_size, num_point, ch)
        # num_point, ch: coord_dim

        # grouped_points (batch_size, k, num_sample, channel)
        # center_points  (batch_size, k, coord_dim)
        grouped_points, center_points = self.sampling_grouping(
            coord_points, feature_points=feature_points)
        # set alias `h` -> (bs, channel, num_sample, k)
        # Note: transpose may be removed by optimizing shape sequence for sampling_groupoing
        h = functions.transpose(grouped_points, (0, 3, 2, 1))
        # h (bs, ch, num_sample_in_region, k=num_group)
        for conv_block in self.feature_extractor_list:
            h = conv_block(h)
        # TODO: try other option of pooling function
        h = functions.max(h, axis=2, keepdims=True)
        # h (bs, ch, 1, k=num_group)
        for conv_block in self.head_list:
            h = conv_block(h)
        h = functions.transpose(h[:, :, 0, :], (0, 2, 1))
        return center_points, h  # (bs, k, coord), h (bs, k, ch')


def _to_array(var):
    """"Input: numpy, cupy array or Variable. Output: numpy or cupy array"""
    if isinstance(var, chainer.Variable):
        var = var.data
    return var


class SamplingGroupingAllModule(chainer.Chain):

    def __init__(self, use_coord=True):
        super(SamplingGroupingAllModule, self).__init__()
        # number of points grouped in each region with radius
        self.use_coord = use_coord

    def __call__(self, coord_points, feature_points=None):
        # input: coord_points (batch_size, num_point, coord_dim)
        # input: feature_points (batch_size, num_point, channel)
        batch_size, num_point, coord_dim = coord_points.shape

        # grouped_points (batch_size, k=1, num_sample, coord_dim)
        grouped_points = coord_points[:, None, :, :]
        # center_points (batch_size, k=1, coord_dim) -> new_coord_points
        center_points = self.xp.zeros((batch_size, 1, coord_dim),
                                      self.xp.float32)

        if feature_points is None:
            new_feature_points = grouped_points
        else:
            # grouped_indices (batch_size, k, num_sample)
            # grouped_feature_points (batch_size, k, num_sample, channel)
            grouped_feature_points = feature_points[:, None, :, :]
            if self.use_coord:
                new_feature_points = functions.concat([grouped_points, grouped_feature_points], axis=3)
            else:
                new_feature_points = grouped_feature_points
        # new_feature_points (batch_size, k, num_sample, channel')
        # center_points (batch_size, k, coord_dim) -> new_coord_points
        return new_feature_points, center_points


if __name__ == '__main__':
    batch_size = 3
    num_point = 100
    coord_dim = 2

    k = 5
    num_sample_in_region = 8
    radius = 0.4
    mlp = [16, 16]
    # mlp2 = [32, 32]
    mlp2 = None

    device = -1
    print('num_point', num_point, 'device', device)
    if device == -1:
        pts = numpy.random.uniform(0, 1, (batch_size, num_point, coord_dim))
    else:
        import cupy
        pts = cupy.random.uniform(0, 1, (batch_size, num_point, coord_dim))

    pts = pts.astype(numpy.float32)
    sam = SetAbstractionGroupAllModule(mlp=mlp, mlp2=mlp2)

    coord, h = sam(pts)
    print('coord', type(coord), coord.shape)  # (3, 5, 2) - (bs, k, coord)
    print('h', type(h), h.shape)              # (3, 5, 32) - (bs, k, ch')
