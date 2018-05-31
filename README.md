# chainer-pointnet

Chainer implementation of PointNet, PointNet++

Original implementation (in tensorflow) can be found on github

 - [charlesq34/pointnet](https://github.com/charlesq34/pointnet)

under MIT license.

## Model structure

`models/pointnet_cls` can be used for classification task.
`trans` option represents to use `TransformNet` or not.
`trans=False` corresponds `PointNetVanilla` (basic),
and `trans=True` corresponds `PointNet` in the paper, respectively.

In my experiment `PointNetVanilla` performs already very well,
the gain in `PoinetNet` is few (maybe only 1-2% gain) while computation becomes
 much huge (around 3 times slower).

## tips

I found the batch normalizations in the last linear layers are quite important.
The accuracy dramatically changes (10% or more) with BatchNormalization at FC
layers.

## Experiments

Experiments in each dataset is located under `expriments` folder.
Each folder is independent, so you can refer independently.

### ModelNet40

This is point cloud classification task of 40 category.
Download script & code is from [charlesq34/pointnet](https://github.com/charlesq34/pointnet)

 - http://modelnet.cs.princeton.edu/

The dataset is automatically downloaded and preprocessed. 
You can simply execute train code to train `PointNet` or `PointNetVanilla`.

```angular2html
# use gpu with id 0, train PointNetVanilla
$ python train.py -g 0 --trans=false

# use gpu with id 0, train PointNet 
$ python train.py -g 0
```

### S3DIS

Stanford Large-Scale 3D Indoor Spaces Dataset (S3DIS) for point cloud segmentation task.
Download the dataset from,

 - [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html)
 
Prerocessing code adopted from [charlesq34/pointnet](https://github.com/charlesq34/pointnet)
under `third_party` directory.

Steps:

1. Go to download link: download S3DIS dataset from
 [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html).
 You need to send application form.
 
2. Download `Stanford3dDataset_v1.2_Aligned_Version.zip` file (4GB),
 place it under `s3dis/data` directory.

2'. Fix mis label manually.

`Stanford3dDataset_v1.2_Aligned_Version/Area_5/hallway_6/Annotations/ceiling_1.txt`
has wrong charcter at line 180389. Please fix it manually.

3. Preprocessing

`collect_indoor3d_data.py` is for data re-organization and 
`gen_indoor3d_h5.py` is to generate HDF5 files. (cite from [charlesq34/pointnet](https://github.com/charlesq34/pointnet/tree/master/sem_seg#dataset))

```angular2html
$ cd third_party
$ python collect_indoor3d_data.py
$ python gen_indoor3d_h5.py
```

4. Training


### ScanNet

Point cloud semantic segmentation task of indoor scenes.

 - [ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes (CVPR 2017 Spotlight)](https://www.youtube.com/watch?v=Olx4OnoZWQQ)

## LICENSE
MIT License.

No warranty or support for this implementation.
Each model performance is not guaranteed, and may not achieve the score reported in each paper. Use it at your own risk.

Please see the [LICENSE](https://github.com/corochann/chainer-pointnet/blob/master/LICENSE) file for details.
