# Multimodal large Language Models for Zero-Shot Human-Object Interaction

## 📄 Paper
This repository contains the code and resources related to the paper:  

**[Full Paper Title]**  
[Author(s)]  
[Conference/Journal Name, Year]  
[Link to paper (arXiv, DOI, or conference page)]  

If you find this work useful, please cite it using the reference below.  

## 📌 Abstract
Multimodal Large Language Models (MLLMs) excel as zero-shot reasoners across diverse domains. However, their application to real-world classification tasks, particularly in direct comparison with specialized models, remains underexplored. This work explores how MLLMs can be leveraged for zero-shot Human-Object Interaction (HOI) recognition and detection using token probability outputs. We first benchmark lightweight MLLMs, identifying Qwen2-VL and MiniCPM-V as the most effective families. We then propose a comprehensive taxonomy of zero-shot strategies, integrating textual prompting methods and visual prompting techniques. For HOI detection, we develop Objects as In-Context Learning (ICL) and Black Other Objects methods for object specification. Evaluations on HICO dataset reveal that Objects as ICL boosts performance for multi-image-capable MLLMs, while ensembling text prompts enhances robustness. On HICO-DET, Objects as ICL, Black Other Objects, and Blur the Background emerge as superior visual prompting methods for localization. Our approach achieves 53.50 mAP on HICO and 23.69 mAP on HICO-DET, outperforming prior zero-shot methods and approaching the performance of certain supervised models.

## 📂 Repository Structure
├── datasets/ # Code to preprocess the data used in the paper
    └── hico
    └── hico_det
├── src/ # Source code for experiments
├── requirements.txt # Dependencies
├── README.md # This file
├── LICENSE # License file

## 🔧 Installation
Clone the repository and install dependencies:
```
git clone https://github.com/oriol-rabasseda/mllms-zs-hoi.git
cd mllms-zs-hoi
pip install -r requirements.txt
```

cd datasets/hico
./build_od.sh /data-slow/hupba/orabasseda/hico_20150920/anno.mat /data-slow/hupba/orabasseda/hico_20150920/images/test2015