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
```
├── datasets/ # Code to preprocess the data used in the paper
    └── hico
    └── hico_det
├── src/ # Source code for experiments
├── requirements.txt # Dependencies
├── README.md # This file
├── LICENSE # License file
```

## 🔧 Installation
Clone the repository and install dependencies:
```
git clone https://github.com/oriol-rabasseda/mllms-zs-hoi.git
cd mllms-zs-hoi
pip install -r requirements.txt
```

## 📥 Dataset Download
The datasets used in this paper can be downloaded from: https://umich-ywchao-hico.github.io/. HICO version is 20150920 and HICO-DET version is 20160224.

After downloading, unzip and place the dataset in `<your-data>` directory.

### HICO pre-processing
Run the script:
```
cd datasets/hico
./build_hico.sh <your-data>/hico_20150920/anno.mat <your-data>/hico_20150920/images/test2015 <mode> <model>
```
where `<mode>` should be substitued by `OD`, `GT`, or `MLLM` depending on the strategy used to filter out questions whose objects are not in the scene:
* `OD`: use YOLO11x to filter the objects in the scene with a confidence score lower than 0.1.
* `GT`: use the ground truth objects in the scene, evaluating the Known Objects approach of HICO.
* `MLLM`: use a MLLM with the question `Is there a <object> in the image? Answer with yes or no.` and compute the classification score to determine the presence of the `<object>` in the scene. The detection score is set to 0.5. This strategy requires the parameter `<model>` with a supported model, that will be used for detection.

To construct the JSON file for the ensemble of prompts, the command is equivalent swapping `./build_hico.sh` for `./build_hico_ensemble.sh` and inserting as 4th argument the number of templates.

### HICO-DET pre-processing
Run the script:
```
cd datasets/hico_det
python possible_questions.py
```

Similar for `possible_questions_bbox`, `possible_questions_blur`, `possible_questions_gray`, and `possible_questions_marker`.

## 🏗️ Usage


## 📊 Results
[Include key results, figures, or tables from the paper.]

## 📜 Citation
If you use this code, please cite:

@article{YourCitationKey,
  author    = {Author Name(s)},
  title     = {Paper Title},
  journal   = {Journal/Conference Name},
  year      = {YYYY},
  volume    = {XX},
  number    = {X},
  pages     = {XX-XX},
  doi       = {DOI or link}
}

## 📜 License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🤝 Acknowledgments
[Mention any funding sources, collaborators, or datasets used.]