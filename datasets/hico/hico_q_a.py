from typing import Dict, List
import json
import numpy as np

def apply_text_prompt_template(object_type: str, action: str, label = int) -> tuple[str, str]:
        object_type_text = object_type.replace('_', ' ')
        action_text = action.replace('_', ' ')
        TEMPLATE = f"Is there anybody {action_text} a {object_type_text}? Answer with yes or no."

        if label == 1.0:
            return TEMPLATE, "yes"
        else:
            return TEMPLATE, "no"


def convert_hico_to_q_a(hico_ann_filepath: str, mode: str):
    with open(hico_ann_filepath, 'r') as f:
        json_ann = json.load(f)

    q_a_items = []
    for img_item in json_ann:  # For each image
        for hoi_instance_one_obj_class in img_item:  # For each HOI in the image
            if hoi_instance_one_obj_class["label"] != 0:
                if mode != 'Unknown':
                    if hoi_instance_one_obj_class["label"] in [-1.0, 1.0]:
                        q_a_items.append(hoi_instance_one_obj_class)
                        hoi_instance_one_obj_class["text_prompt"], hoi_instance_one_obj_class["answer"] = apply_text_prompt_template(
                            hoi_instance_one_obj_class["obj_class"], hoi_instance_one_obj_class["action_gerund"],
                            hoi_instance_one_obj_class["label"])
                else:
                    q_a_items.append(hoi_instance_one_obj_class)
                    hoi_instance_one_obj_class["text_prompt"], hoi_instance_one_obj_class[
                        "answer"] = apply_text_prompt_template(
                        hoi_instance_one_obj_class["obj_class"], hoi_instance_one_obj_class["action_gerund"],
                        hoi_instance_one_obj_class["label"])

    return q_a_items


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hico-ann-filepath", type=str)
    parser.add_argument("--output-qa-filepath", type=str)
    parser.add_argument("--mode", type=str, default="Unknown")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    q_a_items = convert_hico_to_q_a(args.hico_ann_filepath, args.mode)
    with open(args.output_qa_filepath, 'w') as f:
        json.dump(q_a_items, f, indent=4)
