# eSL-Net++





## Environment setup

#### Environment

- Linux
- CUDA 8.0/9.0
- gcc 4.9+
- python 2.7+

#### Dependencies

- pytorch >=1.0.0
- torchvision
- argparse
- numpy
- opencv-python
- scipy

You can create a new [Anaconda](https://www.anaconda.com/products/individual) environment with the above dependencies as follows.
Please make sure to adapt the CUDA toolkit version according to your setup when installing torch and torchvision.

```
conda create -n eslnet python=3.6
conda activate eslnet
conda install pytorch torchvision cudatoolkit -c pytorch
pip install argparse numpy opencv-python scipy 
```

## Download data and pretrained model

#### Data

You can download them via [Google Drive](https://drive.google.com/drive/folders/1ODMevq1aeVuIXCiDpSzEbaJ6cZNowIEe?usp=sharing), which include synthetic testing data (**gopro_test**) from [GoPro dataset](https://seungjunnah.github.io/Datasets/reds.html) and [ESIM](http://rpg.ifi.uzh.ch/esim.html), HQF testing data (**HQF_test**) from [HQF](https://timostoff.github.io/20ecnn) and real data (**realdata_test**)  that we take with DAVIS346.

#### Pretrained model

Pretrained model can be downloaded via [Google Drive](https://drive.google.com/drive/folders/1e7ReB_2no5QWJm4JHloD3365baymTOfd?usp=sharing). 

## Quick start

#### Initialization

- Create directory for testing data

  `mkdir test_data`

  copy the testing data to directory './test_data/'

- Create directory for pretrained model

  `mkdir pre_trained`

  copy the pretrained model to directory './pre_trained/'

- Change the parent directory to './code/'

  `cd code`