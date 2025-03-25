import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm

def main(qa_filepath, aux_qa_filepath, output_filepath) -> None:
    with open(qa_filepath, "r") as f:
        qa_dicts = json.load(f)
    
    with open(aux_qa_filepath, "r") as f:
        aux_qa_dicts = json.load(f)

    obj_per_image = dict()
    for qa_dict in tqdm(aux_qa_dicts, total=len(aux_qa_dicts)):
        if qa_dict['im_path'] not in obj_per_image.keys():
            obj_per_image[qa_dict['im_path']] = [qa_dict['obj_class']]
        else:
            obj_per_image[qa_dict['im_path']] += [qa_dict['obj_class']]

    results = []
    for qa_dict in tqdm(qa_dicts, total=len(qa_dicts)):
        if qa_dict['obj_class'] in obj_per_image[qa_dict['im_path']]:
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
                    "score": 0.0
                })

    # create parent dir for the file output_filepath
    os.makedirs(osp.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w") as f:
        json.dump(results, f, indent=4)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-filepath", type=str)
    parser.add_argument("--aux-qa-filepath", type=str)
    parser.add_argument("--output-filepath", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.qa_filepath, args.aux_qa_filepath, args.output_filepath)
