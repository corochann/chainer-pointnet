"""Used in Kd-Network

Implementation referenced from
https://github.com/fxia22/kdnet.pytorch
kdtree code by @wassname
https://github.com/fxia22/kdnet.pytorch/blob/master/kdtree.py
"""
import scipy.spatial
import numpy


def _parse_split_dims(tree, split_dims, level=0, parent=None, max_level=7,
                      split_positions=None):
    """Traverse KDTree in DFS order, to extract `split_dims`&`split_positions`

    Args:
        tree (scipy.spatial.ckdtree.cKDTreeNode): kdtree node
        split_dims (list): list of list. `split_dims[i]` will store
            `i`-th level `split_dim`.
        level (int): current parsing level
        parent (scipy.spatial.ckdtree.cKDTreeNode):
        max_level (int): max level of KDTree
        split_positions (list): list of list. `split_positions[i]` will store
            `i`-th level `split`. If None, updating this value is skipped.

    """
    if level == max_level:
        # this is leaf tree, and split_dim=-1.
        return

    if tree.lesser is not None:
        _parse_split_dims(tree.lesser, split_dims, level=level+1, parent=tree,
                          max_level=max_level, split_positions=split_positions)
    else:
        # This will happen when the point is overlapped and `split_dim==-1`,
        # in this case just use parent `tree`.
        # print('tree.lesser is None, level {}'.format(level))
        _parse_split_dims(tree, split_dims, level=level+1, parent=tree,
                          max_level=max_level, split_positions=split_positions)
    if tree.greater is not None:
        _parse_split_dims(tree.greater, split_dims, level=level+1, parent=tree,
                          max_level=max_level, split_positions=split_positions)
    else:
        # This will happen when the point is overlapped and `split_dim==-1`,
        # in this case just use parent `tree`.
        # print('[WARNING] tree.greater is None, level {}'.format(level))
        _parse_split_dims(tree, split_dims, level=level+1, parent=tree,
                          max_level=max_level, split_positions=split_positions)

    if level < max_level:
        split_dim = tree.split_dim
        if split_dim == -1:
            # since we repeated premature leafs, we get invalid splits
            # in this case just use the parents.
            # This case happen when the points are overlapped.
            # print('split_dim is -1 at level', level)
            split_dim = parent.split_dim if (parent.split_dim > -1) else 0
        split_dims[level].append(split_dim)
        if split_positions is not None:
            split = tree.split
            if split_dim == -1:
                split = parent.split if (parent.split_dim > -1) else 0
            split_positions[level].append(split)


def calc_max_level(num_point):
    """Calculate estimated max_level based on `num_point`

    Args:
        num_point (int): number of points

    Returns (int): max_level

    """
    return int(numpy.ceil(numpy.log2(num_point)))


def construct_kdtree_data(points, max_level=-1, calc_split_positions=False):
    """Construct preprocessing data for KD-network

    Args:
        points (numpy.ndarray):
            2-dim array (num_point, coord_dim)
        max_level (int): depth of the KDTree. The target size of output points
            is 2**max_level. When -1 is set, minimum target size is
            automatically inferred.
        calc_split_positions (bool): calculate `split_positions` or not.

    Returns:
        new_points (numpy.ndarray):
            2-dim array (num_point', coord_dim), where num_point'=2**max_level
            Its order is updated according to `KDTree.indices`.
        split_dims (numpy.ndarray): list of int array. `split_dims[i]` will
            store `i`-th level `split_dim`.
        inds (numpy.ndarray or slice): 1d array or slice to represent how the
            `new_points` are constructed from input `points`.
        kdtree (scipy.spatial.ckdtree.cKDTree): KDTree instance
        split_positions (numpy.ndarray): list of float array.
            `split_positions[i]` will store `i`-th level `split`.
            If `calc_split_positions=False`, `None` is returned.
            This is mainly for debug purpose.
    """
    assert points.ndim == 2, 'points.ndim must be 2, got points with shape {}'\
        .format(points.shape)
    num_point = points.shape[0]
    if max_level <= -1:
        max_level = calc_max_level(num_point)
    # print('max_level', max_level, 'num_point', num_point)
    target_size = 2 ** max_level
    if target_size > num_point:
        # augment point to make power of 2
        remainder = target_size - num_point
        print('[DEBUG] add points, target_size={}, remainder={}'
              .format(target_size, remainder))
        # select remainder randomly
        inds = numpy.random.choice(range(num_point), remainder)
        inds = numpy.concatenate([numpy.arange(len(points)), inds], axis=0)
        points = points[inds]
    elif target_size < num_point:
        # Reduce number of points
        inds = numpy.random.permutation(num_point)[:target_size]
        points = points[inds]
    else:
        inds = numpy.arange(num_point)
    assert points.shape[0] == target_size
    kdtree = scipy.spatial.cKDTree(points, leafsize=1, balanced_tree=True)
    tree = kdtree.tree
    # split_dims[i] will store `split_dim` for the level `i`.
    split_dims = [[] for _ in range(max_level)]
    if calc_split_positions:
        split_positions = [[] for _ in range(max_level)]
    else:
        split_positions = None
    _parse_split_dims(tree, split_dims, max_level=max_level,
                      split_positions=split_positions)
    split_dims = numpy.array([numpy.array(elem) for elem in split_dims])

    if split_positions is not None:
        # convert list to numpy array with object type.
        split_positions = numpy.array(
            [numpy.array(elem) for elem in split_positions])
    return points[tree.indices], split_dims, inds[tree.indices], kdtree, split_positions


class TransformKDTreeCls(object):

    def __init__(self, max_level=10, return_split_dims=True):
        super(TransformKDTreeCls, self).__init__()
        self.max_level = max_level
        self.return_split_dims = return_split_dims

    def __call__(self, in_data):
        original_points, label = in_data
        # print('original_points', original_points.shape, 'label', label)
        pts = numpy.transpose(original_points[:, :, 0], (1, 0))
        points, split_dims, inds, kdtree, split_positions = construct_kdtree_data(
            pts, max_level=self.max_level,
            calc_split_positions=False)
        points = numpy.transpose(points, (1, 0))[:, :, None]
        if self.return_split_dims:
            # Used in `kdnet_cls`, which needs `split_dims` information.
            return points, split_dims, label
        else:
            # Used in `kdcontextnet_cls`, which only needs permutated points,
            # but not `split_dims`.
            return points, label


class TransformKDTreeSeg(object):

    def __init__(self, max_level=10, return_split_dims=True):
        super(TransformKDTreeSeg, self).__init__()
        self.max_level = max_level
        self.return_split_dims = return_split_dims

    def __call__(self, in_data):
        # shape points (cdim, num_point, 1), label (num_point,)
        original_points, label_points = in_data
        # print('original_points', original_points.shape, 'label', label)
        pts = numpy.transpose(original_points[:, :, 0], (1, 0))
        points, split_dims, inds, kdtree, split_positions = construct_kdtree_data(
            pts, max_level=self.max_level,
            calc_split_positions=False)
        points = numpy.transpose(points, (1, 0))[:, :, None]
        label_points = label_points[inds]
        if self.return_split_dims:
            # Used in `kdnet_cls`, which needs `split_dims` information.
            return points, split_dims, label_points
        else:
            # Used in `kdcontextnet_cls`, which only needs permutated points,
            # but not `split_dims`.
            return points, label_points


if __name__ == '__main__':
    batchsize = 1
    num_point = 135  # try 100, 128, 135
    max_level = 7  # 2^7 -> 128. Final num_point will be 128
    dim = 3
    point_set = numpy.random.rand(num_point, dim)
    print('point_set', point_set.shape)
    points, split_dims, inds, kdtree, split_positions = construct_kdtree_data(
        point_set, max_level=max_level, calc_split_positions=True)
    print('kdtree', kdtree.indices)
    print('inds', inds)
    # print('kdtree.tree', kdtree.tree.indices)  # same with kdtree.indices
    print('points', points.shape)  # 128 point here!
    print(points[0:2])
    print(points[-2:])
    print('split_dims', len(split_dims), 'type', split_dims.dtype)
    print('split_positions', len(split_positions))
    for i in range(len(split_dims)):
        print('i {}, {} type {}'.format(i, split_dims[i], split_dims[i].dtype))
        print('i {}, {} type {}'.format(i, split_positions[i],
                                        split_positions.dtype))

    # test TransformKDTreeSeg
    num_point = 50
    perm = numpy.random.permutation(num_point)
    pts = numpy.arange(num_point).astype(numpy.float32)[perm]
    label = numpy.arange(num_point).astype(numpy.int32)[perm]
    print('pts', pts.shape, pts[:5])
    print('label', label.shape, label[:5])
    pts = numpy.broadcast_to(pts[None, :], (3, num_point))[:, :, None]
    t = TransformKDTreeSeg(max_level=calc_max_level(num_point))
    out_pts, split_dims, out_labels = t((pts, label))
    print('out_pts', out_pts.shape, out_pts[0, :10, 0])
    print('out_labels', out_labels.shape, out_labels[:10])
