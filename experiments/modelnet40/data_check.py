import os

import numpy as np

# data is downloaded when importing provider
import provider


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

    # ModelNet40 official train/test split
    TRAIN_FILES = provider.getDataFiles(
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
    TEST_FILES = provider.getDataFiles(
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

    # these are list of file names for train h5/test h5 files resp.
    print('train_files', TRAIN_FILES)
    print('test_files', TEST_FILES)
