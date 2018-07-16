import os

import h5py

import numpy as np
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


MAX_NUM_POINT = 4096


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def get_dataset(test_area_int=6, num_point=4096):
    assert isinstance(test_area_int, int)
    assert num_point <= MAX_NUM_POINT

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    room_filelist = [line.rstrip() for line in open(
        os.path.join(data_dir, 'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))]

    # Load ALL data
    data_batch_list = []
    label_batch_list = []
    i = 0

    # for h5_filename in all_files:
    while True:
        h5_filename = os.path.join(
            data_dir, 'indoor3d_sem_seg_hdf5_data/ply_data_all_{}.h5'.format(i))
        if not os.path.exists(h5_filename):
            print('exit at i={}'.format(i))
            break
        print('open {}'.format(h5_filename))
        data_batch, label_batch = load_h5(h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
        i += 1

    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    print(data_batches.shape)  # (23585, 4096, 9)  batchsize, num_point, ch
    print(label_batches.shape)  # (23585, 4096)    batchsize, num_point
    # reduce point number num_point
    assert data_batches.ndim == 3
    assert label_batches.ndim == 2
    assert data_batches.shape[0] == label_batches.shape[0]
    assert data_batches.shape[1] == MAX_NUM_POINT
    assert label_batches.shape[1] == MAX_NUM_POINT
    data_batches = data_batches[:, :num_point, :].astype(np.float32)
    label_batches = label_batches[:, :num_point].astype(np.int32)
    # data_batches (batch_size, num_point, k) -> (batch_size, k, num_point, 1)
    data_batches = np.transpose(data_batches, (0, 2, 1))[:, :, :, None]

    # test_area = 'Area_'+str(FLAGS.test_area)
    test_area = 'Area_' + str(test_area_int)
    train_idxs = []
    test_idxs = []
    for i, room_name in enumerate(room_filelist):
        if test_area in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)

    train_data = data_batches[train_idxs, ...]
    train_label = label_batches[train_idxs]
    test_data = data_batches[test_idxs, ...]
    test_label = label_batches[test_idxs]
    print('train shape', train_data.shape, train_label.shape)
    print('test  shape', test_data.shape, test_label.shape)
    train = NumpyTupleDataset(train_data, train_label)
    test = NumpyTupleDataset(test_data, test_label)
    return train, test


if __name__ == '__main__':
    train, test = get_dataset()
    print('train', len(train), 'test', len(test))

    train_x, train_y = train[3]
    print('train_x', train_x.shape, 'train_y', train_y.shape)
    test_x, test_y = test[3]
    print('test_x', test_x.shape, 'test_y', test_y.shape)

    convert_to_kdtree = True
    if convert_to_kdtree:
        from chainer.datasets import TransformDataset
        from chainer_pointnet.utils.kdtree import TransformKDTreeSeg, \
            calc_max_level
        num_point = train_x.shape[1]
        max_level = calc_max_level(num_point)
        train = TransformDataset(train, TransformKDTreeSeg(max_level=max_level))
        points, split_dims, t = train[1]
        print('transformed', points.shape, split_dims.shape, t.shape)
