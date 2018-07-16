import numpy

import chainer
from chainer import functions

from chainer_pointnet.models.conv_block import ConvBlock
from chainer_pointnet.models.pointnet2.set_abstraction_block import \
    SetAbstractionModule
from chainer_pointnet.utils.grouping import _l2_norm


class FeaturePropagationModule(chainer.Chain):

    def __init__(self, mlp, in_channels=None, use_bn=True,
                 activation=functions.relu, residual=False):
        super(FeaturePropagationModule, self).__init__()
        # Feature Extractor channel list
        assert isinstance(mlp, list)
        fe_ch_list = [in_channels] + mlp
        with self.init_scope():
            self.interpolation = InterpolationModule()
            self.feature_extractor_list = chainer.ChainList(
                *[ConvBlock(fe_ch_list[i], fe_ch_list[i+1], ksize=1,
                            use_bn=use_bn, activation=activation,
                            residual=residual
                            ) for i in range(len(mlp))])
        self.use_bn = use_bn

    def __call__(self, distances, points1, points2):
        """

        Args:
            distances (numpy.ndarray or cupy.ndarray):
                3-dim array (bs, num_point2, num_point1)
            points1 (Variable): 3-dim (batch_size, num_point1, ch1)
            points2 (Variable): 3-dim (batch_size, num_point2, ch2)
                points2 is deeper, rich feature. num_point1 > num_point2

        Returns (Variable): 3-dim (batch_size, num_point1, ch1+ch2)
        """
        # h: interpolated_points (batch_size, num_point1, ch1+ch2)
        h = self.interpolation(distances, points1, points2)
        # h: interpolated_points (batch_size, ch1+ch2, num_point1, 1)
        h = functions.transpose(h, (0, 2, 1))[:, :, :, None]
        for conv_block in self.feature_extractor_list:
            h = conv_block(h)
        h = functions.transpose(h[:, :, :, 0], (0, 2, 1))
        return h  # h (bs, num_point, ch')


def _to_array(var):
    """"Input: numpy, cupy array or Variable. Output: numpy or cupy array"""
    if isinstance(var, chainer.Variable):
        var = var.data
    return var


class InterpolationModule(chainer.Chain):

    def __init__(self, num_fp_point=3, eps=1e-10, metric=_l2_norm):
        super(InterpolationModule, self).__init__()
        # number of feature propagation point to interpolate
        self.num_fp_point = num_fp_point
        self.eps = eps
        self.metric = metric

    def __call__(self, distances, points1, points2):
        """

        Args:
            distances (numpy.ndarray or cupy.ndarray):
                3-dim array (bs, num_point2, num_point1)
            points1 (Variable): 3-dim (batch_size, num_point1, ch1)
            points2 (Variable): 3-dim (batch_size, num_point2, ch2)
                points2 is deeper, rich feature. num_point1 > num_point2

        Returns (Variable): 3-dim (batch_size, num_point1, ch1+ch2)

        """
        # batch_size, num_point1, ch1 = points1.shape
        # batch_size2, num_point2, ch2 = points2.shape
        batch_size, num_point2, num_point1 = distances.shape
        # assert batch_size == batch_size2
        if distances is None:
            print('[WARNING] distances is None')
            # calculate distances by feature vector (not coord vector)
            distances = self.xp(self.metric(points1, points2))
            # Better in this form
            # distances = self.xp(self.metric(coord1, coord2))

        # --- weight calculation ---
        # k-nearest neighbor with k=self.num_fp_point
        # sorted_indices (bs, num_fp_point, num_point1)
        sorted_indices = self.xp.argsort(
            distances, axis=1)[:, :self.num_fp_point, :]
        # sorted_dists (bs, num_fp_point, num_point1)
        sorted_dists = distances[
            self.xp.arange(batch_size)[:, None, None],
            sorted_indices,
            self.xp.arange(num_point1)[None, None, :]]

        eps_array = self.xp.ones(
            sorted_dists.shape, dtype=sorted_dists.dtype) * self.eps
        sorted_dists = functions.maximum(sorted_dists, eps_array)
        inv_dist = 1.0 / sorted_dists
        norm = functions.sum(inv_dist, axis=1, keepdims=True)
        norm = functions.broadcast_to(norm, sorted_dists.shape)
        # weight (bs, num_fp_point, num_point1)
        weight = inv_dist / norm
        # --- weight calculation end ---
        # point2_selected (bs, num_fp_point, num_point1, ch2)
        points2_selected = points2[
            self.xp.arange(batch_size)[:, None, None],
            sorted_indices, :]
        # print('debug', weight.shape, points2_selected.shape)
        weight = functions.broadcast_to(
            weight[:, :, :, None], points2_selected.shape)
        # interpolated_points (bs, num_point1, ch2)
        interpolated_points = functions.sum(
            weight * points2_selected, axis=1)
        if points1 is None:
            return interpolated_points
        else:
            return functions.concat([interpolated_points, points1], axis=2)


if __name__ == '__main__':
    batch_size = 4
    num_point1 = 100
    ch1 = 16

    k = 5
    num_sample_in_region = 8
    radius = 0.4

    num_point2 = k
    ch2 = 32
    mlp = [16, ch2]

    device = -1
    print('num_point', num_point1, num_point2, 'device', device)
    if device == -1:
        xp = numpy
    else:
        import cupy
        xp = cupy
    pts = xp.random.uniform(0, 1, (batch_size, num_point1, ch1))

    pts = pts.astype(numpy.float32)
    sam = SetAbstractionModule(k, num_sample_in_region, radius, mlp, mlp2=None,
                               return_distance=True)
    fpm = FeaturePropagationModule([10, 10])

    coord, h, dists = sam(pts)
    h2 = fpm(dists, pts, h)
    print('coord', type(coord), coord.shape)  # (4, 5, 16) - (bs, k, coord)
    print('h', type(h), h.shape)              # (4, 5, 32) - (bs, k, ch')
    print('h2', type(h2), h2.shape)           # (4, 100, 10) - (bs, num_point1, ch')
