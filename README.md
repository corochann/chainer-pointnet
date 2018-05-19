# chainer-pointnet

chainer implementation of PointNet.


Original implementation (in tensorflow) can be found on github

 - [charlesq34/pointnet](https://github.com/charlesq34/pointnet)

under MIT license.

## model structure

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

### ModelNet

This is classification task of 40 category.
Download script & code is from [charlesq34/pointnet](https://github.com/charlesq34/pointnet)

 - http://modelnet.cs.princeton.edu/

### S3DIS

Stanford Large-Scale 3D Indoor Spaces Dataset (S3DIS) for point cloud segmentation task.
Download the dataset from,

 - [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html)
 
Prerocessing code adopted from [charlesq34/pointnet](https://github.com/charlesq34/pointnet)
under `third_party` directory.


### ScanNet

Point cloud semantic segmentation task of indoor scenes.

 - [ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes (CVPR 2017 Spotlight)](https://www.youtube.com/watch?v=Olx4OnoZWQQ)



