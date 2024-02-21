## Develop GAN models using CT images for Inhencing Image Resolution

---

### Project Outline
    - Task
        - Convert CBCT resolution to MDCT resolution
        - Estimate BD value 
    - Aim
        - BD error(difference): +- 50
        - HU error(difference): +- 10 
    - Logging : tensorboard (2.14.0 version)

---

### Model 

- Original Article
  - CycleGAN : https://arxiv.org/pdf/1703.10593.pdf
  - DualGAN : https://openaccess.thecvf.com/content_ICCV_2017/papers/Yi_DualGAN_Unsupervised_Dual_ICCV_2017_paper.pdf

- Architecture:
  - CycleGAN (ResNet based Encoder/Decoder)
    - 3 Encoder Layers, 9 ResBlocks, 3 Decoder Layers  
  - DualGAN (UNET based model)

- Hyperparameters
    - Learning Rate: 2e-4
    - Epoch : 100
    - Batch Size: 2
    - cycle weight: 1e1
    - identity weight: 5e-1
    - Normalization: Instance Normalization

- Deep Learning Framework: 
  - Pytorch (https://pytorch.org/)

---

### Dataset
- Original Image
    - format: DICOM
    - shape: (680, 682) or (700, 702)
    - Number of Images
        - CBCT: 5,971
        - MDCT: 685

- Preprocessing
    - Convertint format (DICOM to tiff)
    - Resize (512, 512)
    - Min Max Scaling (Range: 0~1)

---

### Further
  - Super Resolution (Need Paired Dataset)
  - StyleGAN v2 (Need Code Review)# DualGAN_CT
