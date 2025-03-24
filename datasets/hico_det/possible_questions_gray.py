import json
import numpy as np
import scipy
from tqdm import tqdm
from pyinflect import getInflection

def get_id(obj, action, hico_hoi_list):
    return [hoi for hoi in hico_hoi_list if hoi[1] == obj and hoi[2] == action][0][0]


with open('./hoi_info.json', "r") as f:
    hoi_info = json.load(f)

questions = dict()
for hoi_id in hoi_info.keys():
    hoi_id_info = hoi_info[hoi_id][0]
    object_type_text = hoi_id_info['obj_class'].replace('_', ' ')
    action_text = hoi_id_info['action_gerund'].replace('_', ' ')

    TEMPLATE = f"Is the colored person {action_text} the colored {object_type_text}? Answer yes or no."

    templates = [TEMPLATE]
    
    questions[hoi_id] = {"obj_class": hoi_id_info['obj_class'],
                         "action": hoi_id_info['action'],
                         "questions": templates}

with open('./annotations/questions_per_hoi_gray.json', "w") as new_jsonl_file:
    json.dump(questions, new_jsonl_file, indent=4)
