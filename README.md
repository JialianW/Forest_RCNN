# ACM MM 2020 Anonymous Submission: Forest R-CNN: Large-Vocabulary Long-Tailed Object Detection and Instance Segmentation

This repo is a official implementation of "Forest R-CNN: Large-Vocabulary Long-Tailed Object Detection and Instance Segmentation"based on open-mmlab's mmdetection. Many thanks to mmdetection for their simple and clean framework. If you have any questions, please comment in the issue and we will reply asap.

Models are available for downloading~~~

## Installation 
Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Train and inference
The Forest R-CNN config is in [configs/lvis](configs/lvis).

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
    
    
    
# Main Results
 
## Instance Segmentation on LVIS

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
<td align="center"><a href="https://www.dropbox.com/s/r6iw74udaon8yas/forest_rcnn_res50.pth?dl=0">model</a>&nbsp</td>
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
<td align="center"><a href="https://www.dropbox.com/s/el6xd1gr3p6xyai/forest_rcnn_res101.pth?dl=0">model</a>&nbsp</td>
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
<td align="center"><a href="https://www.dropbox.com/s/4txz2nu1vnmlrqf/forest_rcnn_resnext101.pth?dl=0">model</a>&nbsp</td>
</tr>

</tbody></table>

Due to the space limit for free account in Dropbox, we only release the models of our method.





