# VerdaSense-ML

This repository contains the implementation scripts for wound segmentation using **MobileSAM with LoRA Fine-tuning**. 

---

## 🌐 Live Demo
Experience the model in action without any setup:
- **HuggingFace Space:** [Wound Segmentation & Tissue Classification](https://huggingface.co/spaces/BP17/WoundSegmenter)
  - *Note: Simply upload a wound image, provide the bounding box(es), and get the segmented results and tissue overlays.*

---

## 📂 Directory Structure

### 1. [data/](./data)
**Original Dataset Sources:**
* **DFUC 2022:** [Diabetes Foot Ulcer Challenge 2022](https://www.kaggle.com/datasets/pabodhamallawa/dfuc2022-train-release)
* **FUSeg:** [Foot Ulcer Segmentation Challenge](https://github.com/uwm-bigdata/wound-segmentation/blob/master/data/Foot%20Ulcer%20Segmentation%20Challenge/README.MD/)

Contains the cleaned and split datasets (1019 Training, 391 Validation).
- `train.zip` / `validation.zip`: The primary split.
- `test-FUSeg.zip` / `test-DFUC2022.zip`: Individual test sets for benchmarking.

### 2. [training/](./training)
- `[FineTuneFromScratch]LoRAFinetuning_MobileSAM.ipynb`: LoRA Finetuning script of MobileSAM.
- `[LoadFromCheckpoints]LoRAFinetuning_MobileSAM.ipynb`: Use this to resume training from the checkpoints.
- `MobileSAMwithDomainAdpater.ipynb`: Training script of MobileSAM with a small learnable domain adapter module.
- `DeepLabV3+withMobileNetV2.ipynb`: Training script of DeepLabV3+ with MobileNetV2 backbone.

### 3. [weights/](./weights)
- `best_finetunedMobileSAM_DA.pth`: Weights for the Domain Adapter version.
- `mask_decoder.pth` & `lora_image_encoder.zip`: The essential weights for the **LoRA** model.
- `best_DeepLabv3PlusModelwithMobileNetV2.pth`: Weights of DeepLabV3+ with MobileNetV2 backbone.

### 4. [evaluation/](./evaluation)
- Scripts to calculate **Dice Score** and **IoU**.
- `EvaluationOnRealWorldDataset.ipynb`: Performance of **Average Symmetric Surface Distance (ASSD)** on real-world collected images.

### 5. [inference/](./inference)
- `[Inference]LoRA_Finetuned_MobileSAM.ipynb`: Run this to get the segmentation masks only.
- `[Inference]LoRA_Finetuned_MobileSAM_withKMeans.ipynb`: Run this to get both the segmentation masks and tissue classification overlays.

---

## 🛠️ Usage & Environment Setup

This project was developed and tested on **Google Colab** using the **T4 GPU** runtime. 

To run these scripts on your own account:
1. **Hardware:** Set your runtime to **T4 GPU** (required for faster training).
2. **Data Setup:** - Upload the `.zip` files from the `/data` directory to your Google Drive.
   - The notebooks are configured to unzip these files within the Colab VM for optimal I/O speed.
3. **Path Configuration:** - Locate the `path` variables (clearly marked in each notebook).
   - Update these variables to point to your specific Google Drive directory paths.

---

## 📚 References & Literature

This project builds upon the following research papers:

* **DeepLabV3+:** [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611)
* **SAM:** [Segment Anything](https://arxiv.org/pdf/2304.02643)
* **MobileSAM:** [Faster Segment Anything: Towards Lightweight SAM for Mobile Applications](https://arxiv.org/pdf/2306.14289)
* **SAM-DA:** [SAM-DA: Decoder Adapter for Efficient Medical Domain Adaptation](https://arxiv.org/html/2501.06836v1)