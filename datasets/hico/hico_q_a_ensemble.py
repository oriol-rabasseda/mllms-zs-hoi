from typing import Dict, List
import json
import numpy as np
from pyinflect import getInflection
from tqdm import tqdm

PERSON_OPTIONS = ['Are the people in the image ', 'Are the individuals in the image ', 'Are the humans in the image ',
                  'Are the people ', 'Are the individuals ', 'Are the humans ']

np.random.seed(0)

def apply_text_prompt_template(object_type: str, action: str, label: int, num_templates: int, synonyms):
    object_type_text = object_type.replace('_', ' ')
    action_text = action.replace('_', ' ')

    TEMPLATE = f"Is there anybody {action_text} a {object_type_text}? Answer with yes or no."

    if action != 'no_interaction':
        action_synonyms_gerund = []
        for act in synonyms:
            act_words = act.split('_')
            if len(act_words) == 1:
                word = act
            else:
                word = act_words[0]
            
            action_gerund = getInflection(word, 'VBG')

            if action_gerund != None:
                if len(act_words) == 1:
                    action_synonyms_gerund.append(action_gerund[0])
                else:
                    action_synonyms_gerund.append(' '.join([action_gerund[0]] + act_words[1:]))
        
        action_synonyms_gerund = list(set(action_synonyms_gerund + [action_text]))
    
        all_templates = [f"{init}{action} a {object_type_text}? Answer yes or no." for action in action_synonyms_gerund for init in PERSON_OPTIONS]
        
        templates = [TEMPLATE]
        if num_templates-1 < len(all_templates):
            templates += np.random.choice(all_templates, num_templates-1, replace=False).tolist()
        else:
            templates += all_templates
            templates = np.resize(templates, num_templates).tolist()
        
    else:
        templates = [TEMPLATE] * num_templates

    if label == 1.0:
        return templates, "yes"
    else:
        return templates, "no"


def convert_hico_to_q_a(hico_ann_filepath: str, hoi_info_filename: str, mode: str, num_templates: int):
    with open(hico_ann_filepath, 'r') as f:
        json_ann = json.load(f)
    
    with open(hoi_info_filename, 'r') as f:
        hoi_info = json.load(f)

    q_a_items = []
    for img_item in tqdm(json_ann, total=len(json_ann)):  # For each image
        for hoi_instance_one_obj_class in img_item:  # For each HOI in the image
            if hoi_instance_one_obj_class["label"] != 0:
                if mode != 'Unknown':
                    if hoi_instance_one_obj_class["label"] in [-1.0, 1.0]:
                        q_a_items.append(hoi_instance_one_obj_class)
                        hoi_instance_one_obj_class["text_prompt"], hoi_instance_one_obj_class["answer"] = apply_text_prompt_template(
                            hoi_instance_one_obj_class["obj_class"], hoi_instance_one_obj_class["action_gerund"],
                            hoi_instance_one_obj_class["label"], num_templates, 
                            get_synonyms(hoi_info, hoi_instance_one_obj_class["action"], hoi_instance_one_obj_class["obj_class"]))
                else:
                    q_a_items.append(hoi_instance_one_obj_class)
                    hoi_instance_one_obj_class["text_prompt"], hoi_instance_one_obj_class[
                        "answer"] = apply_text_prompt_template(
                        hoi_instance_one_obj_class["obj_class"], hoi_instance_one_obj_class["action_gerund"],
                        hoi_instance_one_obj_class["label"], num_templates,
                        get_synonyms(hoi_info, hoi_instance_one_obj_class["action"], hoi_instance_one_obj_class["obj_class"]))

    return q_a_items

def get_synonyms(hoi_info, action, obj):
    hoi_id = [hoi_id for hoi_id in hoi_info.keys() if hoi_info[hoi_id][0]['obj_class'] == obj and hoi_info[hoi_id][0]['action'] == action][0]
    
    action_synonyms = []
    for synset in hoi_info[hoi_id]:
        action_synonyms += synset['action_synonyms'].split(',')
    
    return action_synonyms


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hico-ann-filepath", type=str)
    parser.add_argument("--output-qa-filepath", type=str)
    parser.add_argument("--mode", type=str, default="Unknown")
    parser.add_argument("--num-templates", type=int, default=1)
    parser.add_argument("--hoi-info-filepath", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    q_a_items = convert_hico_to_q_a(args.hico_ann_filepath, args.hoi_info_filepath, args.mode, args.num_templates)
    with open(args.output_qa_filepath, 'w') as f:
        json.dump(q_a_items, f, indent=4)
