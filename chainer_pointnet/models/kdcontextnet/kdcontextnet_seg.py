import numpy

import chainer
from chainer import functions
from chainer import links
from chainer import reporter

from chainer_pointnet.models.conv_block import ConvBlock
from chainer_pointnet.models.kdcontextnet.kdcontextconv_block import \
    KDContextConvBlock
from chainer_pointnet.models.kdcontextnet.kdcontextdeconv_block import \
    KDContextDeconvBlock


class KDContextNetSeg(chainer.Chain):

    """Segmentation 3DContextNet

    Args:
        out_dim (int): output dimension, number of class for classification
        in_dim (int or None): input dimension for each point.
            default is 3, (x, y, z).
        dropout_ratio (float): dropout ratio
        use_bn (bool): use batch normalization or not.
        compute_accuracy (bool): compute & report accuracy or not
        levels (list): list of int. It determines block depth and each
            block's grouping (receptive field) size.
            For example if `list=[5, 6, 7]`, it will take
            level1 block as 5-th depth (32) of KDTree,
            level2 as 6-th depth (64), level3 as 7-th depth (128) resp.
            Receptive field (number of points in each grouping) is
            `2**level` for each level.
        feature_learning_mlp_enc_list (list): list of list of int.
            Indicates each block's feature learning MLP size for encoder.
        feature_aggregation_mlp_enc_list (list): list of list of int.
            Indicates each block's feature aggregation MLP size for encoder.
        feature_learning_mlp_dec_list (list): list of list of int.
            Indicates each block's feature learning MLP size for decoder.
        feature_aggregation_mlp_dec_list (list): list of list of int.
            Indicates each block's feature aggregation MLP size for decoder.
        normalize (bool): apply normalization to calculate global context cues
            in `KDContextConvBlock` & KDContextDeconvBlock.
        residual (bool): use residual connection or not
    """

    def __init__(self, out_dim, in_dim=3, dropout_ratio=-1,
                 use_bn=True, compute_accuracy=True,
                 levels=None,
                 feature_learning_mlp_enc_list=None,
                 feature_aggregation_mlp_enc_list=None,
                 feature_learning_mlp_dec_list=None,
                 feature_aggregation_mlp_dec_list=None,
                 fc_mlp_list=None, normalize=False, residual=False
                 ):
        super(KDContextNetSeg, self).__init__()
        levels = levels or [5, 7, 9]  # (32, 128, 512) receptive field.
        if feature_learning_mlp_enc_list is None:
            feature_learning_mlp_enc_list = [
                [64, 64, 128, 128], [64, 64, 256, 256], [64, 64, 512, 512]]
        if feature_aggregation_mlp_enc_list is None:
            feature_aggregation_mlp_enc_list = [[256], [512], [1024]]
        if feature_learning_mlp_dec_list is None:
            feature_learning_mlp_dec_list = list(reversed(
                feature_learning_mlp_enc_list))
        if feature_aggregation_mlp_dec_list is None:
            feature_aggregation_mlp_dec_list = list(reversed(
                feature_aggregation_mlp_enc_list))
        fc_mlp_list = fc_mlp_list or [128]

        in_channels_enc_list = [in_dim] + [
            elem[-1] for elem in feature_aggregation_mlp_enc_list]
        in_channels_dec_list = [in_channels_enc_list[-1]] + [
            elem[-1] for elem in feature_aggregation_mlp_dec_list]
        levels_diff = numpy.diff(numpy.array([0] + levels))
        print('levels {}, levels_diff {}'.format(levels, levels_diff))
        fcmlps = [feature_aggregation_mlp_dec_list[-1][-1]] + fc_mlp_list
        assert len(levels) == len(feature_learning_mlp_enc_list)
        assert len(levels) == len(feature_aggregation_mlp_enc_list)
        assert len(levels) == len(feature_learning_mlp_dec_list)
        assert len(levels) == len(feature_aggregation_mlp_dec_list)
        with self.init_scope():
            # don't use dropout in conv blocks
            self.kdcontextconv_blocks = chainer.ChainList(
                *[KDContextConvBlock(
                    in_channels_enc_list[i], m=2 ** levels_diff[i],
                    feature_learning_mlp=feature_learning_mlp_enc_list[i],
                    feature_aggregation_mlp=feature_aggregation_mlp_enc_list[i],
                    use_bn=use_bn, normalize=normalize, residual=residual
                ) for i in range(len(levels_diff))])
            self.kdcontextdeconv_blocks = chainer.ChainList(
                *[KDContextDeconvBlock(
                    in_channels_dec_list[i], m=2 ** levels_diff[-i-1],
                    out_deconv_channels=in_channels_dec_list[i]//2,
                    feature_learning_mlp=feature_learning_mlp_dec_list[i],
                    feature_aggregation_mlp=feature_aggregation_mlp_dec_list[i],
                    use_bn=use_bn, normalize=normalize, residual=residual
                ) for i in range(len(levels_diff))])
            self.conv_blocks = chainer.ChainList(
                *[ConvBlock(
                    fcmlps[i], fcmlps[i+1], ksize=1, use_bn=use_bn,
                    residual=residual
                ) for i in range(len(fcmlps)-1)])
            self.conv = links.Convolution2D(fcmlps[-1], out_dim, ksize=1)
        self.compute_accuracy = compute_accuracy
        self.dropout_ratio = dropout_ratio

    def calc(self, x):
        # x (bs, ch, N, 1)
        assert x.ndim == 4
        assert x.shape[3] == 1
        h = x
        h_skip_list = [h]
        for kdconv_block in self.kdcontextconv_blocks:
            h = kdconv_block(h)
            h_skip_list.append(h)
        # h (bs, ch, N//2**levels[-1], 1)
        # TODO: support other symmetric function
        # TODO: Support concatenating whole global feature.
        # h = functions.max_pooling_2d(h, ksize=(h.shape[2], 1))

        h_skip_list.pop(-1)  # don't use last h as skip connection.
        for kddeconv_block in self.kdcontextdeconv_blocks:
            h_skip = h_skip_list.pop(-1)
            h = kddeconv_block(h, h_skip)
        assert len(h_skip_list) == 0
        for conv_block in self.conv_blocks:
            h = conv_block(h)
        if self.dropout_ratio > 0.:
            h = functions.dropout(h, self.dropout_ratio)
        h = self.conv(h)
        return h[:, :, :, 0]

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
    from chainer_pointnet.utils.kdtree import construct_kdtree_data

    batchsize = 1
    num_point = 135  # try 100, 128, 135
    max_level = 9  # 2^7 -> 128. Final num_point will be 128
    dim = 3
    point_set = numpy.random.rand(num_point, dim).astype(numpy.float32)
    print('point_set', point_set.shape)
    points, split_dims, inds, kdtree, split_positions = construct_kdtree_data(
        point_set, max_level=max_level, calc_split_positions=True)
    print('points', points.shape)  # 128 point here!
    kdnet = KDContextNetSeg(out_dim=5, use_bn=False)
    # split_dims = numpy.array(split_dims)
    # print('split_dims', split_dims.shape, split_dims.dtype)
    pts = numpy.transpose(points, (1, 0))[None, :, :, None]
    print('pts', pts.shape, split_dims.shape)
    out = kdnet.calc(pts)
    print('out', out.shape)




