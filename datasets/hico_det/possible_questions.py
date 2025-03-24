import json
import numpy as np
import scipy
from tqdm import tqdm
from pyinflect import getInflection

def get_id(obj, action, hico_hoi_list):
    return [hoi for hoi in hico_hoi_list if hoi[1] == obj and hoi[2] == action][0][0]

PERSON_OPTIONS = ['Are the people in the image ', 'Are the individuals in the image ', 'Are the humans in the image ',
                  'Are the people ', 'Are the individuals ', 'Are the humans ']

with open('./hoi_info.json', "r") as f:
    hoi_info = json.load(f)

questions = dict()
for hoi_id in hoi_info.keys():
    hoi_id_info = hoi_info[hoi_id][0]
    object_type_text = hoi_id_info['obj_class'].replace('_', ' ')
    action_text = hoi_id_info['action_gerund'].replace('_', ' ')

    TEMPLATE = f"Is there anybody {action_text} a {object_type_text}? Answer with yes or no."

    templates = [TEMPLATE]
    if hoi_id_info['action'] != 'no_interaction':
        for init in PERSON_OPTIONS:
            action_synonyms_gerund = []
            for act in hoi_id_info['action_synonyms'].split(','):
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

            for act_syn in action_synonyms_gerund:
                new_template = f"{init}{act_syn} a {object_type_text}? Answer yes or no."
                templates.append(new_template)
    
    questions[hoi_id] = {"obj_class": hoi_id_info['obj_class'],
                         "action": hoi_id_info['action'],
                         "questions": templates}

with open('./annotations/questions_per_hoi.json', "w") as new_jsonl_file:
    json.dump(questions, new_jsonl_file, indent=4)
