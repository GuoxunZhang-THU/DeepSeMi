# Deep learning based self-supervised enhanced microscopy
<img src="img/logo_DeepSeMi.jpg" width="800" align="center">
Implementation for deep learning based self-supervised enhanced microscopy (DeepSeMi)

![Imaging modality](https://img.shields.io/badge/Imaging%20modality-Fluorescence%20microscopy-brightgreen)  ![Purpose](https://img.shields.io/badge/Purpose-Video%20denoising-orange)  

## 📋 Table of content
    1. [Overview](#Overview)
 2. [Quick start DeepSeMi](#Start)
    1. [Environment](#Environment)
    2. [Install dependencies](#Dependencies)
    3. [Download the demo code and data](#Download)
    4. [Run the trained model](#Run)
    5. [Work for your own data](#Owndata)
   3. [Other information](#Information)
    1. [Results](#Results)
    2. [Citation](#Citation)
    2. [Email](#Email)

## **📚** Overview <a name="Overview"></a>
Fluorescence microscopy provides optical access to comprehensive longitudinal live cells multi-spectral imaging, which assists biological scientists discover numerous new biological phenomenas. We developed a deep learning based self-supervised enhanced microscopy (DeepSeMi) established on a novel 3D full blind spot convolution, which is able to recover low SNR recordings without high SNR recordings.  For more details and results, please see our companion paper titled 

<img src="img/figure_concept_sub.png" width="400" align="center">

## **⏳** Quick start DeepSeMi <a name="Start"></a>
This tutorial will show how DeepSeMi enhances the  microscopic captures.
### **💡** Environment <a name="Environment"></a>
* Ubuntu 16.04 
* Python 3.6
* Pytorch >= 1.3.1
* NVIDIA GPU (24 GB Memory) + CUDA

### **💡** Install dependencies <a name="Dependencies"></a>
* Create a virtual environment and install some dependencies.
```
$ conda create -n deepsemi_env python=3.6
$ source activate deepsemi_env
$ pip install -q torch==1.10.0
$ pip install -q torchvision==0.8.2
$ pip install deepsemi
$ pip install -q opencv-python==4.1.2.30
$ pip install -q tifffile  
$ pip install -q scikit-image==0.17.2
$ pip install -q scikit-learn==0.24.1
```
### **💡** Download the demo code and data <a name="Download"></a>
```
$ git clone git://github.com/yuanlong-o/Deep_widefield_cal_inferece
$ cd DeepCAD/DeepWonder/
```
We upload a demo data on Google drive: [low light intensity mitochondria imaging by the confocal microscopy](https://drive.google.com/drive/folders/1WiTrL5gRuMUssMYt2uDRDO-5pmmrdNSc?usp=sharing). To run the demo script, those data need to be downloaded and put into the *DeepSeMi/datasets* folder. We upload a trained denoising model and you can download it into *DeepSeMi/pth* folder. 


### **💡** Run the trained model <a name="Run"></a>
Run the script.py to enhance the demo data. The necessary prompt information is written in the *script.py*.
```
$ python script.py
```
The output from the demo script can be found in the *DeepSeMi/results* folder. 
### **💡** Work for your own data <a name="Owndata"></a>
Run the script.py to retrain the network based on your own data. You need to put your data in *DeepSeMi/datasets* folder. Then follow the tutorial information in the *script.py* to run the code. The intermediate results of code running will be saved under *DeepSeMi/results*. You can utilize them to monitor the training process.

```
$ python script.py
```

## 🤝 Other information <a name="Information"></a>
### **📝** Results

Some of our results are exhibited below. For more results and further analyses, please refer to the companion paper where this method first occurred.

**Confocal imaging of cell migration**

<img src="img/results1.png" width="800" align="center">

### **📝** Citation <a name="Citation"></a>

If you use this code and relevant data, please cite the corresponding paper where original methods appeared: https://www.biorxiv.org/content/10.1101/2022.11.02.514874v1


### **📝** Email <a name="Email"></a>
We are pleased to address any questions regarding the above tools through emails (zhanggx19@mails.tsinghua.edu.cn ).
