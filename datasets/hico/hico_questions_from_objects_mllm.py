import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(osp.abspath(sys.path[0] + '/../../src'))

from models import Qwen2VLModel
from models.minicpm_v import MiniCPM_V
from models.llava_onevision_model import LLaVAOneVisionModel
from models.phi_model import PhiVisionModel
from models.internvl2 import InternVL2Model
from models.ovis2_model import Ovis2Model
from models.deepseekvl2 import DeepseekVL2Model
from models.base_model import BaseModel
from Constants import *


def main(model_name, img_dir, qa_filepath, output_filepath) -> None:
    with open(qa_filepath, "r") as f:
        qa_dicts = json.load(f)

    model = init_model(model_name)
    results = []
    current_file = ""
    counter = 0
    for qa_dict in tqdm(qa_dicts, total=len(qa_dicts)):
        img_filepath = osp.join(img_dir, qa_dict["im_path"])

        if current_file != img_filepath:
            current_file = img_filepath
            detected = list()
            det_scores = dict()

            for obj in COCO_OBJECTS.values():
                prompt = f"Is there a {obj} in the image? Answer with yes or no."
                _, _, score = model.infer(prompt, img_filepath)

                if score >= 0.5:
                    detected.append(obj)
                
                det_scores[obj] = score

        if qa_dict['obj_class'] in detected:
            results.append(
                {
                    "im_path": qa_dict["im_path"],
                    "text_prompt": qa_dict["text_prompt"],
                    "obj_class": qa_dict["obj_class"],
                    "action": qa_dict["action"],
                    "model_output": "",
                    "answer": qa_dict["answer"],
                    "score": -1.0
                })
        
        else:
            results.append(
                {
                    "im_path": qa_dict["im_path"],
                    "text_prompt": "",
                    "obj_class": qa_dict["obj_class"],
                    "action": qa_dict["action"],
                    "model_output": "",
                    "answer": qa_dict["answer"],
                    "score": det_scores[qa_dict["obj_class"]]
                })

    # create parent dir for the file output_filepath
    os.makedirs(osp.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w") as f:
        json.dump(results, f, indent=4)


def init_model(model_name) -> BaseModel:
    if "minicpm-v" in model_name.lower():
        return MiniCPM_V(model_name)
    elif "qwen" in model_name.lower():
        return Qwen2VLModel(model_name)
    elif "llava-onevision" in model_name.lower():
        return LLaVAOneVisionModel(model_name)
    elif "internvl2" in model_name.lower():
        return InternVL2Model(model_name)
    elif "phi" in model_name.lower():
        return PhiVisionModel(model_name)
    elif "ovis" in model_name.lower():
        return Ovis2Model(model_name)
    elif "deepseek-vl2" in model_name.lower():
        return DeepseekVL2Model(model_name)
    else:
        raise NotImplementedError

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--img-dir", type=str)
    parser.add_argument("--qa-filepath", type=str)
    parser.add_argument("--output-filepath", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.model_name, args.img_dir, args.qa_filepath, args.output_filepath)
