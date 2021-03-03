# Forest R-CNN: Large-Vocabulary Long-Tailed Object Detection and Instance Segmentation (ACM MM 2020)

Official implementation of:

[Forest R-CNN: Large-Vocabulary Long-Tailed Object Detection and Instance Segmentation](https://arxiv.org/abs/2008.05676)
\
Jialian Wu, Liangchen Song, Tiancai Wang, Qian Zhang and Junsong Yuan
\
In ACM International Conference on Multimedia , Seattle WA, October 12-16, 2020.

Many thanks to [mmdetection](https://github.com/open-mmlab/mmdetection) authors for their great framework!

## News
Mar 2, 2021 Update:
We test Forest R-CNN on LVIS v1.0 set. Thanks for considering comparing with our method :)

Jan 1, 2021 Update: We propose Forest DetSeg, an extension of original Forest R-CNN. Forest DetSeg extends the proposed method to RetinaNet.
While the new work is under review now, the code has been available. More details will come up along with the new paper.

## Installation
Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.

## Forest R-CNN
### Inference
    # Examples
    # single-gpu testing
    python tools/test.py configs/lvis/forest_rcnn_r50_fpn.py forest_rcnn_res50.pth --out out.pkl --eval bbox segm
    
    # multi-gpu testing
    ./tools/dist_test.sh configs/lvis/forest_rcnn_r50_fpn.py forest_rcnn_res50.pth ${GPU_NUM} --out out.pkl --eval bbox segm

### Training
    # Examples
    # single-gpu training
    python tools/train.py configs/lvis/forest_rcnn_r50_fpn.py --validate
    
    # multi-gpu training
    ./tools/dist_train.sh configs/lvis/forest_rcnn_r50_fpn.py ${GPU_NUM} --validate
(Note that we found in our experiments the best result comes up around the 20-th epoch instead of the end of training.)

## Forest RetinaNet
### Inference
    # Examples  
    # multi-gpu testing
    ./tools/dist_test.sh configs/lvis/forest_retinanet_r50_fpn_1x.py forest_retinanet_res50.pth ${GPU_NUM} --out out.pkl --eval bbox segm

### Training
    # Examples    
    # multi-gpu training
    ./tools/dist_train.sh configs/lvis/forest_retinanet_r50_fpn_1x.py ${GPU_NUM} --validate

# Main Results

## Instance Segmentation on LVIS v0.5 val set

AP and AP.b denote the mask AP and box AP. r, c, f represent the rare, common, frequent contegoires.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom", align="left">Method</th>
<th valign="bottom", align="left">Backbone</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP.r</th>
<th valign="bottom">AP.c</th>
<th valign="bottom">AP.f</th>
<th valign="bottom">AP.b</th>
<th valign="bottom">AP.b.r</th>
<th valign="bottom">AP.b.c</th>
<th valign="bottom">AP.b.f</th>
<th valign="bottom">download</th>

<!-- TABLE BODY -->
<tr>
<td align="left">MaskRCNN</td>
<td align="left">R50-FPN</td>
<td align="center">21.7</td>
<td align="center">6.8</td>
<td align="center">22.6</td>
<td align="center">26.4</td>
<td align="center">21.8</td>
<td align="center">6.5</td>
<td align="center">21.6</td>
<td align="center">28.0</td>
<td align="center"><a href="https://drive.google.com/file/d/1kSzY8moyGEettNKecmnW0wKDXrcvqmRZ/view?usp=sharing">model</a>&nbsp</td>
</tr>

<tr>
<td align="left">Forest R-CNN</td>
<td align="left">R50-FPN</td>
<td align="center">25.6</td>
<td align="center">18.3</td>
<td align="center">26.4</td>
<td align="center">27.6</td>
<td align="center">25.9</td>
<td align="center">16.9</td>
<td align="center">26.1</td>
<td align="center">29.2</td>
<td align="center"><a href="https://drive.google.com/file/d/1UJJjDpweSV5YwxYWhkbk5fweYgwD8awK/view?usp=sharing">model</a>&nbsp</td>
</tr>

<tr>
<td align="left">MaskRCNN</td>
<td align="left">R101-FPN</td>
<td align="center">23.6</td>
<td align="center">10.0</td>
<td align="center">24.8</td>
<td align="center">27.6</td>
<td align="center">23.5</td>
<td align="center">8.7</td>
<td align="center">23.1</td>
<td align="center">29.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1P3Za3hPBUYGLA-5WJZbZRdi9DTsPr5A7/view?usp=sharing">model</a>&nbsp</td>
</tr>
<tr>
<td align="left">Forest R-CNN</td>
<td align="left">R101-FPN</td>
<td align="center">26.9</td>
<td align="center">20.1</td>
<td align="center">27.9</td>
<td align="center">28.3</td>
<td align="center">27.5</td>
<td align="center">20.0</td>
<td align="center">27.5</td>
<td align="center">30.4</td>
<td align="center"><a href="https://drive.google.com/file/d/1f3qwbl2yFrKWrFQx8KMme5WkRoGJiknI/view?usp=sharing">model</a>&nbsp</td>
</tr>

<tr>
<td align="left">MaskRCNN</td>
<td align="left">X-101-32x4d-FPN</td>
<td align="center">24.8</td>
<td align="center">10.0</td>
<td align="center">26.4</td>
<td align="center">28.6</td>
<td align="center">24.8</td>
<td align="center">8.6</td>
<td align="center">25.0</td>
<td align="center">30.9</td>
<td align="center"><a href="https://drive.google.com/file/d/1Gon4Wwk2Mh7tJmNpze5NS0w6opjNxl-G/view?usp=sharing">model</a>&nbsp</td>
</tr>
<tr>
<td align="left">Forest R-CNN</td>
<td align="left">X-101-32x4d-FPN</td>
<td align="center">28.5</td>
<td align="center">21.6</td>
<td align="center">29.7</td>
<td align="center">29.7</td>
<td align="center">28.8</td>
<td align="center">20.6</td>
<td align="center">29.2</td>
<td align="center">31.7</td>
<td align="center"><a href="https://drive.google.com/file/d/1qxQesUHv6snxeF5IzhvUBSTBO26jCNI1/view?usp=sharing">model</a>&nbsp</td>
</tr>

</tbody></table>

## Instance Segmentation on LVIS v1.0 val set
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom", align="left">Method</th>
<th valign="bottom", align="left">Backbone</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP.r</th>
<th valign="bottom">AP.c</th>
<th valign="bottom">AP.f</th>
<th valign="bottom">AP.b</th>

<!-- TABLE BODY -->
<tr>
<td align="left">MaskRCNN</td>
<td align="left">R50-FPN</td>
<td align="center">19.2</td>
<td align="center">0.0</td>
<td align="center">17.2</td>
<td align="center">29.5</td>
<td align="center">20.0</td>
</tr>

<tr>
<td align="left">Forest R-CNN</td>
<td align="left">R50-FPN</td>
<td align="center">23.2</td>
<td align="center">14.2</td>
<td align="center">22.7</td>
<td align="center">27.7</td>
<td align="center">24.6</td>
</tr>
</tbody></table>
## Visualized Examples
![](examples.png)


## Citation
If you find it useful in your research, please consider citing our paper as follows:

    @inproceedings{wu2020forest,
    title={Forest R-CNN: Large-vocabulary long-tailed object detection and instance segmentation},
    author={Wu, Jialian and Song, Liangchen and Wang, Tiancai and Zhang, Qian and Yuan, Junsong},
    booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
    pages={1570--1578},
    year={2020}}





