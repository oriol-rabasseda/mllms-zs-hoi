import __init__
import vsrl_utils as vu
from vsrl_eval import VCOCOeval
import pickle
import numpy as np
import json
from possible_questions import gerunds

# Load COCO annotations for V-COCO images
coco = vu.load_coco()

# Load the VCOCO annotations for vcoco_train image set
vcoco_all = vu.load_vcoco('vcoco_trainval')
for x in vcoco_all:
    x = vu.attach_gt_boxes(x, coco)

for x in vcoco_all:
    x['action_name'] = x['action_name'].decode()
    x['role_name'] = [i.decode() for i in x['role_name']]
    x['include'] = [[j.decode() for j in i] for i in x['include']]

possible_hoi = []

for vcoco in vcoco_all:
    for i in range(len(vcoco['image_id'])):
        role_bbox = vcoco['role_bbox'][i,:]
        role_bbox = role_bbox.reshape((-1,4))
        obj, instr = '', ''
        for j in range(1, len(vcoco['role_name'])):
            if not np.isnan(role_bbox[j, 0]):
                if vcoco['role_name'][j] == 'obj': obj = coco.loadCats([coco.loadAnns(ids=[vcoco['role_object_id'][i][j]])[0]['category_id']])[0]['name']
                if vcoco['role_name'][j] == 'instr': instr = coco.loadCats([coco.loadAnns(ids=[vcoco['role_object_id'][i][j]])[0]['category_id']])[0]['name']
            
        hoi = {'action': vcoco['action_name'],
               'obj': obj,
               'instr': instr,
               'action_gerund': gerunds[vcoco['action_name']]
               }
        
        if obj == '' and instr == '':
            question = f"Is the colored person {hoi['action_gerund']}? Answer yes or no."
        elif obj == '':
            question = f"Is the colored person {hoi['action_gerund']} with the colored {hoi['instr']}? Answer with yes or no."
        elif instr == '':
            question = f"Is the colored person {hoi['action_gerund']} the colored {hoi['obj']}? Answer with yes or no."
        else:
            question = f"Is the colored person {hoi['action_gerund']} the colored {hoi['obj']} with the colored {hoi['instr']}? Answer with yes or no."
        
        hoi['question'] = question

        if hoi not in possible_hoi:
            possible_hoi.append(hoi)

with open("annotations/questions_per_hoi_gray.json", 'w') as f:
    json.dump(possible_hoi, f, indent=4)