# SegNet

SegNet is a model of semantic segmentation based on Fully Comvolutional Network.

This repository contains the implementation of learning and testing in keras and tensorflow.
Also included is a custom layer implementation of index pooling, a new property of segnet.

![architectire](https://user-images.githubusercontent.com/27678705/33704504-199ba3ea-db70-11e7-8009-dc23aa9770a0.png)

## architecture
- encoder decoder architecture
- fully convolutional network
- indices pooling

    ![indecespooling](https://user-images.githubusercontent.com/27678705/33704612-81053eec-db70-11e7-9822-01dd48d68314.png)

## description
This repository is SegNet architecture for Semantic Segmentation.
The repository of other people's segmentation, pooling with indices not implemented.But In this repository we implemented  pooling layer and unpooling layer with indices at MyLayers.py.

Segnet architecture is early Semantic Segmentation model,so acccuracy is low but fast.
In the future, we plan to implement models with high accuracy.(UNet,PSPNet,Pix2Pix ect..)



## Usage

### train

`python SegNet.py [--options your dataset]`

## DEMO
- dataset
  - LIP(Look Into Person)

<div align="center">
<img src=https://user-images.githubusercontent.com/27678705/32144033-0e57b3f4-bcf6-11e7-89fe-737e98db5f6d.png title="oroginal" width="200px"><img src=https://user-images.githubusercontent.com/27678705/32144037-1c5cae32-bcf6-11e7-9834-f1b1b13b535c.png title="ground truth" width="200px"><img src=https://user-images.githubusercontent.com/27678705/32144070-a5cbecb4-bcf6-11e7-8de9-af3c9b68fa7f.png title="predict" width="200px">
</div>

The problem of binary mask works well with SegNet.
<div align="center">
<img src="https://user-images.githubusercontent.com/27678705/33703457-8a504fdc-db6b-11e7-8922-db3c61294b18.png" alt="demo2">
</div>
