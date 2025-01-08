<h1 align="center">Enhanced Medical Image Segmentation via Deep Dynamic Self-Adjusting U-Net with Multi-scale Attention and Semantic Mitigation</h1>
<p align="center">Yanchi Ou, Yufeng Chen, Yuzhi Zhang, Shukai Yang, Xiaoqian Zhang, Ying Zhou, Siyu Chen and Lifan Peng</p>

## News
Our following work "Enhanced Medical Image Segmentation via Deep Dynamic Self-Adjusting U-Net with Multi-scale Attention and Semantic Mitigation" is being submitted to The Visual Computer.

## Abstract
Convolutional neural networks (CNNs) and Vision Transformers (ViTs) have demonstrated remarkable performance in medical image segmentation. However, existing approaches often suffer from limitations such as ignoring the interconnection between feature channels and spaces, inadequate contextual information extraction, and semantic differences in skip connections. To address these issues, we propose a Deep Dynamic Self-Adjusting U-Net (DDS-UNet) equipped with a Multi-scale Self-attention Mechanism (MSM) and a Semantic Mitigation Module (SMM). The MSM effectively extracts multi-scale and global feature information, while the SMM bridges the semantic gap between encoder and decoder layers. Furthermore, we introduce a Lightweight Deformable Residual module (LDR) to replace the standard convolution in UNet, enabling efficient and accurate contour feature extraction. Experimental results on four medical image datasets demonstrate the superiority of DDS-UNet in terms of segmentation accuracy, computational efficiency, and robustness. Our approach achieves state-of-the-art performance, highlighting its potential for practical medical image segmentation tasks.

## Network

![image](https://github.com/user-attachments/assets/ed149df3-cf29-4fd8-b8f4-818488190b1b)

Overall block diagram of DDS-UNet. In the UNet encoder part, an LDR module is constructed to dynamically capture shallow contour information. A simple and effective SMM is proposed to alleviate the semantic gap. The MSM module which only needs 1/4 channel feature to extract full scale information is designed.

![image](https://github.com/user-attachments/assets/d2562134-4d1a-480f-96c0-33b67f23adfe)

Results of qualitative comparison of DDS-UNet with other methods.


## Segmentation results of DDS-UNet
### Qualitative result



![image](https://github.com/user-attachments/assets/a103f7f4-5bf6-4127-af18-0b750df8febf)


### Quantitative result
<p align="center">Performance comparison of DDS-UNet with other baseline methods on the CVC-ClinicDB dataset.</p>

![image](https://github.com/user-attachments/assets/f3e5f5fd-5b60-413b-97dd-1ed4cda40a3b)

<p align="center">Performance comparison of DDS-UNet with other baseline methods on the Kvasir-SEG dataset.</p>

![image](https://github.com/user-attachments/assets/1b8ef92b-8b0e-4fe7-991a-ef87917d529f)

<p align="center">Performance comparison of DDS-UNet with other baseline methods on ISIC2018 dataset.</p>

![image](https://github.com/user-attachments/assets/72db370c-dc00-4b9b-8a48-e29ab4cb0456)

<p align="center">Performance comparison of DDS-UNet with other baseline methods on Breast Ultrasound Dataset B.</p>

![image](https://github.com/user-attachments/assets/f769767b-e96d-450f-bc7e-504b678d878f)


## Getting Started
### Environment
1.Clone this repo:https://github.com/ououyy/DDS-UNet.git

2.Create a new conda environment and install dependencies:

pip:
```
    - addict==2.4.0
    - dataclasses==0.8
    - mmcv-full==1.2.7
    - numpy==1.19.5
    - opencv-python==4.5.1.48
    - perceptual==0.1
    - pillow==8.4.0
    - scikit-image==0.17.2
    - scipy==1.5.4
    - tifffile==2020.9.3
    - timm==0.3.2
    - torch==1.7.1
    - torchvision==0.8.2
    - typing-extensions==4.0.0
    - yapf==0.31.0
```
### Training & Test
Python train.py

Python val.py

### Datasets
CVC-ClinicDB:
 https://polyp.grand-challenge.org/CVCClinicDB/

Kvasir-SEG: 
https://datasets.simula.no/kvasir-seg/

ISIC2018: 
https://challenge.isic-archive.com/data/#2018

Breast Ultrasound Dataset B: https://helward.mmu.ac.uk/STAFF/M.Yap/dataset.php

## Citation
```
@article{ou2025enhanced,
  title={Enhanced Medical Image Segmentation via Deep Dynamic Self-Adjusting U-Net with Multi-scale Attention and Semantic Mitigation},
  author={Yanchi Ou, Yufeng Chen, Yuzhi Zhang, Shukai Yang, Xiaoqian Zhang, Ying Zhou, Siyu Chen and Lifan Peng},
  journal={The Visual Computer},
  year={2025},
  publisher={Springer}
}
```
## Acknowledgment
A part of this code is adapted from the previous work: UNeXt (https://github.com/jeya-maria-jose/UNeXt-pytorch).
