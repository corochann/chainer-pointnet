import numpy

import chainer
from chainer import functions
from chainer import links

from chainer_pointnet.models.conv_block import ConvBlock
from chainer_pointnet.models.linear_block import LinearBlock
from chainer_pointnet.utils.grouping import query_ball_by_diff
from chainer_pointnet.utils.sampling import farthest_point_sampling


class SetAbstractionModule(chainer.Chain):

    def __init__(self, k, num_sample_in_region, radius,
                 mlp, mlp2,
                 in_channels=None, use_bn=True,
                 activation=functions.relu):
        # k is number of sampled point (num_region)
        super(SetAbstractionModule, self).__init__()
        # Feature Extractor channel list
        assert isinstance(mlp, list)
        fe_ch_list = [in_channels] + mlp
        # Head channel list
        assert isinstance(mlp2, list)
        head_ch_list = [mlp[-1]] + mlp2
        with self.init_scope():
            self.sampling_grouping = SamplingGroupingModule(
                k=k, num_sample_in_region=num_sample_in_region, radius=radius)
            self.feature_extractor_list = chainer.ChainList(
                *[ConvBlock(fe_ch_list[i], fe_ch_list[i+1], ksize=1,
                            use_bn=use_bn, activation=activation
                            ) for i in range(len(mlp))])
            self.head_list = chainer.ChainList(
                *[ConvBlock(head_ch_list[i], head_ch_list[i + 1], ksize=1,
                            use_bn=use_bn, activation=activation
                            ) for i in range(len(mlp2))])
        self.use_bn = use_bn

    def __call__(self, coord_points, feature_points=None):
        # coord_points   (batch_size, ch, k)
        # TODO: transpose `coord`

        # grouped_points (batch_size, k, num_sample, channel)
        #TODO: better to have (bs, channel, num_sample, k) shape sequence
        grouped_points, center_points = self.sampling_grouping(
            coord_points, feature_points=feature_points)
        # set alias `h`
        # TODO: transpose will be removed after shape sequence is optimized for sampling_groupoing
        h = functions.transpose(grouped_points, (0, 3, 2, 1))
        # h (bs, ch, num_sample_in_region, k=num_group)
        for conv_block in self.feature_extractor_list:
            h = conv_block(h)
        h = functions.max(h, axis=2)
        # h (bs, ch, 1, k=num_group)
        for conv_block in self.head_list:
            h = conv_block(h)
        return h


#TODO: this does not have parameter, change it to function instead of chain.
class SamplingGroupingModule(chainer.Chain):

    def __init__(self, k, num_sample_in_region, radius=None, use_coord=True):
        super(SamplingGroupingModule, self).__init__()
        # number of center point (sampled point)
        self.k = k

        # number of points grouped in each region with radius
        self.radius = radius
        self.num_sample_in_region = num_sample_in_region
        self.use_coord = use_coord

    def __call__(self, coord_points, feature_points=None):
        # input: coord_points (batch_size, num_point, coord_dim)
        # input: feature_points (batch_size, num_point, channel)
        batch_size, num_point, coord_dim = coord_points.shape

        # sampling
        farthest_indices, distances = farthest_point_sampling(
            coord_points, self.k, skip_initial=True)
        # grouping
        grouped_indices = query_ball_by_diff(
            distances, self.num_sample_in_region, radius=self.radius)

        # grouped_indices (batch_size, k, num_sample)
        # grouped_points (batch_size, k, num_sample, coord_dim)
        grouped_points = coord_points[self.xp.arange(batch_size)[:, None, None], grouped_indices, :]
        # center_points (batch_size, k, coord_dim) -> new_coord_points
        center_points = coord_points[self.xp.arange(batch_size)[:, None], farthest_indices, :]

        # calculate relative coordinate
        grouped_points = grouped_points - center_points[:, :, None, :]

        # TODO: concat grouped_points & feature_points to get new_feature_points
        if feature_points is None:
            new_feature_points = grouped_points
        else:
            # grouped_indices (batch_size, k, num_sample)
            # grouped_feature_points (batch_size, k, num_sample, channel)
            grouped_feature_points = feature_points[
                                     self.xp.arange(batch_size)[:, None, None],
                                     grouped_indices, :]
            if self.use_coord:
                new_feature_points = functions.concat([grouped_points, grouped_feature_points], axis=3)
            else:
                new_feature_points = grouped_feature_points
        return new_feature_points, center_points
