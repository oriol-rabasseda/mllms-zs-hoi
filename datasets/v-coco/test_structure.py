import __init__
import vsrl_utils as vu
from vsrl_eval import VCOCOeval
import pickle
import numpy as np
import json

# Load COCO annotations for V-COCO images
coco = vu.load_coco()

# Load the VCOCO annotations for vcoco_train image set
vcoco_all = vu.load_vcoco('vcoco_test')
for x in vcoco_all:
    x = vu.attach_gt_boxes(x, coco)

for x in vcoco_all:
    x['action_name'] = x['action_name'].decode()
    x['role_name'] = [i.decode() for i in x['role_name']]
    x['include'] = [[j.decode() for j in i] for i in x['include']]

results = []
for i in range(len(vcoco_all[0]['image_id'])):
    coco_image = coco.loadImgs(ids=[vcoco_all[0]['image_id'][i][0]])[0]
    results.append({'image_id': vcoco_all[0]['image_id'][i][0],
                    'person_box': vcoco_all[0]['bbox'][[i],:][0],
                    'im_path': coco_image['file_name'],
                    })

for vcoco in vcoco_all:
    for i in range(len(vcoco['image_id'])):
        results[i][f"{vcoco['action_name']}_agent"] = vcoco['label'][i][0]
        results[i][f"{vcoco['action_name']}_obj"] = [0, 0, 0, 0, vcoco['label'][i][0]]
        results[i][f"{vcoco['action_name']}_instr"] = [0, 0, 0, 0, vcoco['label'][i][0]]
        
        role_bbox = vcoco['role_bbox'][i,:]
        role_bbox = role_bbox.reshape((-1,4))
        for j in range(1, len(vcoco['role_name'])):
            if not np.isnan(role_bbox[j, 0]):
                results[i][f"{vcoco['action_name']}_{vcoco['role_name'][j]}"] = list(role_bbox[j, :]) + [vcoco['label'][i][0]]

with open("annotations/test.pickle", "wb") as output_file:
    pickle.dump(results, output_file)