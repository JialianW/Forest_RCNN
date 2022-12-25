## Installation
This instruction is orignated from the original mmdetection. Many thanks to mmdetection for their simple and clean framework!
### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+ (Python 2 is not supported)
- PyTorch 1.1 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC(G++) 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC(G++): 4.9/5.3/5.4/7.3

### Install Forest R-CNN

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/).


c. Clone the Forest R-CNN repository.

```shell
git clone https://github.com/JialianW/Forest_RCNN.git
cd Forest_RCNN
```

d. Install Forest_RCNN (other dependencies will be installed automatically).

```shell
pip install -r requirements.txt
pip install mmcv
python setup.py develop  # or "pip install -v -e ."
```

### Install the C++ implementation of NMS-Resampling

In order to make use of the faster implementation of NMS-Resampling in C++, it is required to install `mmcv` from source. Please note that currently only the linear version is implemented in C++.

In this explanation we will use `mmcv` version `1.6.0`, but the process can be easily adapted to newer versions. 

The following commands **should replace** the third line of the step "d" in the previous section (`pip install mmcv`).

```shell
git clone --depth 1 --branch v1.6.0 https://github.com/open-mmlab/mmcv.git
rsync -a mmcv_custom/ ./mmcv
cd mmcv
pip install -r requirements/optional.txt
pip install -e .
```

### Prepare datasets

The images of LVIS are the same as those of COCO. Note that the LVIS version has been updated to v1.0, 
while our experiments are conducted based on v0.5. We therefore provide the annotations of the LVIS v0.5 that can be found [here](https://drive.google.com/drive/folders/1SEnLiexLW-sE_PAnzz5pXm1iaBe8Aj_u?usp=sharing).
The LVIS v1.0 is [here](https://www.lvisdataset.org/dataset).
```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── lvis
│   │   ├── annotations
│   │   ├── train2017/000000000009.jpg...
│   │   ├── val2017/000000000139.jpg...




