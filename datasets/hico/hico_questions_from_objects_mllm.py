import os
import os.path as osp
import json

import sys
sys.path.append(osp.abspath(sys.path[0] + '/../'))

from qwen2_vl_model import Qwen2VLModel
from minicpm_v import MiniCPM_V
from llava_onevision_model import LLaVAOneVisionModel
from phi_model import PhiVisionModel
from llava_next_model import LLaVANextModel
from internvl2 import InternVL2Model
from base_model import BaseModel
import numpy as np
from Constants import *

from tqdm import tqdm

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
            
            if qa_dict["action"] != "no_interaction":
                counter += 1
        
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
    
    print(counter)


def init_model(model_name) -> BaseModel:
    if "minicpm-v" in model_name.lower():
        return MiniCPM_V(model_name)
    elif "qwen2-vl" in model_name.lower():
        return Qwen2VLModel(model_name)
    elif "llava-onevision" in model_name.lower():
        return LLaVAOneVisionModel(model_name)
    elif "internvl2" in model_name.lower():
        return InternVL2Model(model_name)
    elif "phi" in model_name.lower():
        return PhiVisionModel(model_name)
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
