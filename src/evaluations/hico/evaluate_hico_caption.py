import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(osp.abspath(sys.path[0] + '/../..'))

from models import Qwen2VLModel
from models.minicpm_v import MiniCPM_V
from models.base_model import BaseModel
from Constants import *

def main(model_name, img_dir, qa_filepath, output_filepath) -> None:
    with open(qa_filepath, "r") as f:
        qa_dicts = json.load(f)

    model = init_model(model_name)
    results = []
    current_file = ""
    for qa_dict in tqdm(qa_dicts, total=len(qa_dicts)):
        img_filepath = osp.join(img_dir, qa_dict["im_path"])

        if current_file != img_filepath:
            cumulative_score = 1
            current_file = img_filepath

            description, vision_hidden_states, _ = model.infer(CAPTION_QUESTION,
                                                     img_filepath,
                                                     vision_hidden_states=None,
                                                     max_new_tokens = CAPTION_MAX_TOKENS)
            if description[-1] != '.':
                    description += '.'

        if qa_dict['score'] == -1:
            if qa_dict['action'] != 'no_interaction':
                new_prompt = "Context: " + description + ' Question: ' + qa_dict["text_prompt"]

                model_output, _, score = model.infer(new_prompt,
                                                        img_filepath,
                                                        vision_hidden_states=vision_hidden_states,
                                                        max_new_tokens = 1)
                
                cumulative_score *= 1-score

                results.append(
                    {
                        "im_path": qa_dict["im_path"],
                        "text_prompt": new_prompt,
                        "obj_class": qa_dict["obj_class"],
                        "action": qa_dict["action"],
                        "model_output": model_output,
                        "answer": qa_dict["answer"],
                        "score": score
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
                        "score": cumulative_score
                    })
                cumulative_score = 1
        
        else:
            results.append(
                {
                    "im_path": qa_dict["im_path"],
                    "text_prompt": "",
                    "obj_class": qa_dict["obj_class"],
                    "action": qa_dict["action"],
                    "model_output": "",
                    "answer": qa_dict["answer"],
                    "score": qa_dict["score"]
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
