import os

import h5py

import numpy as np
from chainer_chemistry.datasets import NumpyTupleDataset


def get_data_files(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def get_dataset(test_area_int=6):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    # all_files = get_data_files(
    #     os.path.join(data_dir, 'indoor3d_sem_seg_hdf5_data/all_files.txt'))

    # from glob import glob
    # all_files = glob(
    #     os.path.join(data_dir, 'indoor3d_sem_seg_hdf5_data/ply_data_all_0.h5'))
    # print('allfiles', all_files)
    # import IPython; IPython.embed()

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
    print(data_batches.shape)
    print(label_batches.shape)
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
