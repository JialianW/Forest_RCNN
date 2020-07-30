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
git clone https://github.com/annonymous1212/Forest_RCNN.git
cd Forest_RCNN
```

d. Install Forest_RCNN (other dependencies will be installed automatically).

```shell
pip install -r requirements.txt
pip install mmcv
python setup.py develop  # or "pip install -v -e ."
```




### Prepare datasets

Let's prepare the dataset of LVIS. The images of LVIS are the same as those of COCO. The images and annotations can be download [here](https://www.lvisdataset.org/dataset).
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




