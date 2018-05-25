import numpy
from chainer import cuda

from chainer_pointnet.utils.sampling import farthest_point_sampling


def _l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.

    Args:
        x (numpy.ndarray or cupy.ndarray):
            input points, 3-dim array (batch_size, num_point, coord_dim)
        y (numpy.ndarray or cupy.ndarray):
            query points, 3-dim array (batch_size, k, coord_dim)

    Returns (numpy.ndarray): (batch_size, k, num_point,)

    """
    return ((x[:, None, :, :] - y[:, :, None, :]) ** 2).sum(axis=3)


def query_ball_point(pts, indices, num_sample, radius=None, metrics=_l2_norm):
    """query ball point

    `num_sample` nearest neighbor search for each `indices` among `pts`.
    When `radius` is set, query point must be less than `radius`,

    Args:
        pts (numpy.ndarray or cupy.ndarray): input points
            3-dim array (batch_size, num_point, coord_dim)
        indices: 2-dim array (batch_size, k) indices of `pts` to be center.
        num_sample (int): number of points selected in each region
        radius (float or None): radius of each region to search.
            When `None`, it is equivalent to the case radius is infinity,
            and the behavior is same k-nearest neighbor with `k=num_sample`.
        metrics (callable): metrics function to calculate distance

    Returns (numpy.ndarray or cupy.ndarray): grouped indices
            3-dim array (batch_size, k, num_sample)

    """
    # --- calc diff ---
    diff = calc_diff(pts, indices, metrics=metrics)
    # --- sort & select nearest indices ---
    return query_ball_by_diff(diff, num_sample=num_sample, radius=radius)


def calc_diff(pts, indices, metrics=_l2_norm):
    batch_size = pts.shape[0]
    # query_pts (batch_size, k, coor_dim) -> k points to be center
    query_pts = pts[numpy.arange(batch_size)[:, None], indices, :]

    # diff (batch_size, k, num_point)
    diff = metrics(pts, query_pts)
    return diff


def query_ball_by_diff(diff, num_sample, radius=None):
    """

    Args:
        diff:
        num_sample:
        radius:

    Returns (numpy.ndarray or cupy.ndarray): grouped indices
            3-dim array (batch_size, k, num_sample)

    """
    xp = cuda.get_array_module(diff)
    diff_sorted_indices = xp.argsort(diff, axis=2)
    if radius is None:
        return diff_sorted_indices[:, :, :num_sample]
    else:
        diff_sorted_indices = diff_sorted_indices[:, :, :num_sample]
        diff_sorted = diff[xp.arange(batch_size)[:, None, None],
                           xp.arange(k)[None, :, None],
                           diff_sorted_indices]
        # take original value when it is smaller than radius.
        return xp.where(diff_sorted < radius, diff_sorted_indices,
                        diff_sorted_indices[:, :, 0:1])


if __name__ == '__main__':
    # when num_point = 10000 & k = 1000 & batch_size = 32,
    # CPU takes 6 sec, GPU takes 0.5 sec.

    from contextlib import contextmanager
    from time import time

    @contextmanager
    def timer(name):
        t0 = time()
        yield
        t1 = time()
        print('[{}] done in {:.3f} s'.format(name, t1-t0))

    # batch_size = 8
    # num_point = 10000
    # coord_dim = 2
    # k = 1000
    # do_plot = False
    batch_size = 3
    num_point = 200
    coord_dim = 2
    k = 8
    do_plot = False

    # for grouping
    num_sample = 10
    # radius = None
    # radius = 10.0
    radius = 0.0000001

    device = -1
    print('num_point', num_point, 'device', device)
    if device == -1:
        pts = numpy.random.uniform(0, 1, (batch_size, num_point, coord_dim))
    else:
        import cupy
        pts = cupy.random.uniform(0, 1, (batch_size, num_point, coord_dim))

    with timer('farthest_point_sampling'):
        farthest_indices, distances = farthest_point_sampling(
            pts, k, skip_initial=True)
    print('farthest_indices', farthest_indices.shape, type(farthest_indices))

    # efficient calculation using `distances`
    with timer('query_ball_by_diff'):
        grouped_indices = query_ball_by_diff(distances, num_sample, radius=radius)
    print('grouped_indices', grouped_indices.shape, type(grouped_indices))
    # (batch_size, k, num_sample)

    # query_ball_point, it will calculate `diff` <- same with `distances` above,
    # takes time for calculation
    with timer('query_ball_point'):
        grouped_indices2 = query_ball_point(pts, farthest_indices, num_sample, radius=radius)
    print('grouped_indices2', grouped_indices.shape, type(grouped_indices))
    assert numpy.allclose(cuda.to_cpu(grouped_indices),
                          cuda.to_cpu(grouped_indices2))

    # take grouped points
    grouped_points = pts[numpy.arange(batch_size)[:, None, None], grouped_indices, :]
    print('grouped_points', grouped_points.shape)
    for i in range(grouped_points.shape[0]):
        for j in range(grouped_points.shape[1]):
            for k in range(grouped_points.shape[2]):
                index = grouped_indices[i, j, k]
                numpy.allclose(grouped_points[i, j, k, :], pts[i, index, :])
                # print('test grouped_points',
                #       numpy.sum(grouped_points[i, j, k, :] - pts[i, index, :]))

    if do_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os
        pts = cuda.to_cpu(pts)
        farthest_indices = cuda.to_cpu(farthest_indices)
        grouped_indices_flatten = cuda.to_cpu(
            grouped_indices).reshape(batch_size, k * num_sample)
        if not os.path.exists('results'):
            os.mkdir('results')
        for index in range(batch_size):
            fig, ax = plt.subplots()
            plt.grid(False)
            plt.scatter(pts[index, :, 0], pts[index, :, 1], c='k', s=4)
            # farthest point itself is also inside grouped point, so write grouped points before drawing farthest points.
            plt.scatter(pts[index, grouped_indices_flatten[index], 0], pts[index, grouped_indices_flatten[index], 1], c='b', s=4)
            plt.scatter(pts[index, farthest_indices[index], 0], pts[index, farthest_indices[index], 1], c='r', s=4)
            # plt.show()
            plt.savefig('results/farthest_point_sampling_grouping_{}.png'.format(index))
