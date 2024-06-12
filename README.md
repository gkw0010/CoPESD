# CoPESD: A Multi-Level Surgical Motion Dataset for Training Large Vision-Language Models to Co-Pilot Endoscopic Submucosal Dissection
Guankun Wang∗, Han Xiao∗, Huxin Gao, Renrui Zhang, Long Bai, Xiaoxiao Yang, Zhen Li, Hongsheng Li†, Hongliang Ren†

## Overview
With the advances in surgical robotics, robot-assisted endoscopic submucosal dissection (ESD) enables rapid resection of large lesions, minimizing recurrence rates and improving long-term overall survival. Despite these advantages, ESD is technically challenging and carries high risks of complications, necessitating skilled surgeons and precise instruments. Recent advancements in Large Visual-Language Models (LVLMs) offer promising decision support and predictive planning capabilities for robotic systems, which can augment the accuracy of ESD and reduce procedural risks. However, existing datasets for multi-level fine-grained ESD surgical motion understanding are scarce and lack detailed annotations. In this paper, we design a hierarchical decomposition of ESD motion granularity and introduce a multi-level surgical motion dataset (CoPESD) for training LVLMs as the robotic Co-Pilot of Endoscopic Submucosal Dissection. CoPESD includes 17,679 images with 32,699 bounding boxes and 88,395 multi-level motions, from over 35 hours of ESD videos for both robot-assisted and conventional surgeries. CoPESD enables granular analysis of ESD motions, focusing on the complex task of submucosal dissection. Extensive experiments on the LVLMs demonstrate the effectiveness of CoPESD in training LVLMs to predict following surgical robotic motions. As the first multimodal ESD motion dataset, CoPESD supports advanced research in ESD instruction-following and surgical automation.

## Features

- CoPESD is built based on **a granular decomposition of surgical motions**, providing precise motion definitions for ESD.
- CoPESD is a fine-grained multi-level surgical motion dataset including **17,679 images with 32,699 bounding boxes and 88,395 multi-level motions**.
- We provide the link to download CoPESD.

<p align="center">
  <img 
    width="1000"
    src="./figures/overview.png"
  >
</p>


