import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import cv2
import math
import pickle
import itertools

import sys
sys.path.append(osp.abspath(sys.path[0] + '/../..'))

from models import Qwen2VLModel
from models.minicpm_v import MiniCPM_V
from models.llava_onevision_model import LLaVAOneVisionModel
from models.phi_model import PhiVisionModel
from models.internvl2 import InternVL2Model
from models.ovis2_model import Ovis2Model
#from models.deepseekvl2 import DeepseekVL2Model
from models.base_model import BaseModel
from Constants import *
from evaluations.hico_det.evaluate_hico_det import min_factor, ksize, angle, startAngle, endAngle

def get_ids(objects, hoi_info):
    hois = [h for h in hoi_info if h['obj'] == "" and h['instr'] == ""]
    new_objects = {obj[0] for obj in objects}
    
    for obj in new_objects:
        hois += [h for h in hoi_info if h['obj'] == obj and h['instr'] == ""]
        hois += [h for h in hoi_info if h['obj'] == "" and h['instr'] == obj]        

    for comb in list(itertools.combinations(new_objects, 2)):
        hois += [h for h in hoi_info if h['obj'] == comb[0] and h['instr'] == comb[1]]
        hois += [h for h in hoi_info if h['obj'] == comb[1] and h['instr'] == comb[0]]
        hois += [h for h in hoi_info if h['obj'] == comb[0] and h['instr'] == comb[0]]
        hois += [h for h in hoi_info if h['obj'] == comb[1] and h['instr'] == comb[1]]

    hoi_list = []
    for hoi in hoi_info:
        if hoi in hois:
            if hoi['instr'] != "" and hoi['obj']:
                hoi_list_aux = []
                for obj in [o for o in objects if o[0] == hoi['obj']]:
                    for instr in [o for o in objects if o[0] == hoi['instr']]:
                        new_hoi = hoi
                        new_hoi[f"{hoi['action']}_agent"] = -1
                        new_hoi[f"{hoi['action']}_obj"] = obj[1]
                        new_hoi[f"{hoi['action']}_instr"] = instr[1]
                        hoi_list_aux.append(new_hoi)
                hoi_list.append(hoi_list_aux)
            elif hoi['instr'] != "":
                hoi_list_aux = []
                for instr in [o for o in objects if o[0] == hoi['instr']]:
                    new_hoi = hoi
                    new_hoi[f"{hoi['action']}_agent"] = -1
                    new_hoi[f"{hoi['action']}_obj"] = [0, 0, 0, 0, 0]
                    new_hoi[f"{hoi['action']}_instr"] = instr[1]
                    hoi_list_aux.append(new_hoi)
                hoi_list.append(hoi_list_aux)
            elif hoi['obj'] != "":
                hoi_list_aux = []
                for obj in [o for o in objects if o[0] == hoi['obj']]:
                    new_hoi = hoi
                    new_hoi[f"{hoi['action']}_agent"] = -1
                    new_hoi[f"{hoi['action']}_obj"] = obj[1]
                    new_hoi[f"{hoi['action']}_instr"] = [0, 0, 0, 0, 0]
                    hoi_list_aux.append(new_hoi)
                hoi_list.append(hoi_list_aux)
            else:
                new_hoi = hoi
                new_hoi[f"{hoi['action']}_agent"] = -1
                new_hoi[f"{hoi['action']}_obj"] = [0, 0, 0, 0, 0]
                new_hoi[f"{hoi['action']}_instr"] = [0, 0, 0, 0, 0]
                hoi_list.append([new_hoi])
        else:
            new_hoi = hoi
            new_hoi[f"{hoi['action']}_agent"] = 0
            new_hoi[f"{hoi['action']}_obj"] = [0, 0, 0, 0, 0]
            new_hoi[f"{hoi['action']}_instr"] = [0, 0, 0, 0, 0]
            hoi_list.append([new_hoi])
    
    return hoi_list

def main(model_name, img_dir, gt_filepath, q_hoi_filepath, output_filepath, mode, save_images_path):   
    with open(q_hoi_filepath, "r") as f:
        hoi_info = json.load(f)
    
    with open(gt_filepath, "rb") as f:
        image_ids = pickle.load(f)
        image_ids = list({(x['image_id'], x['im_path']) for x in image_ids})
    
    obj_detector = YOLO("yolo11x.pt").to('cuda:0')
    model = init_model(model_name)

    #results = []
    with open(output_filepath, "rb") as f:
        results = pickle.load(f)
    
    for (im_id, im_path) in tqdm(image_ids, total=len(image_ids)):
        im_done = list({res['image_id'] for res in results})
        if im_id in im_done:
            continue
        img_filepath = osp.join(img_dir, im_path)
        image = Image.open(img_filepath).convert("RGB")
        detections = obj_detector(image, conf=0.5, verbose=False)[0]
        image_count = 0
        
        detected_objects = []
        detected_person = []
        for obj in enumerate(detections.boxes.data):
            x1, y1, x2, y2 = obj[1][0].item(), obj[1][1].item(), obj[1][2].item(), obj[1][3].item()
            obj_type = detections.names[obj[1][5].item()]
            if obj_type == 'person':
                size = (x2-x1)*(y2-y1)
                detected_person.append((size, [x1, y1, x2, y2]))
            
            detected_objects.append((obj_type, [x1, y1, x2, y2], obj[1][4].item()))
        
        selected_objects = sorted(detected_objects, key= lambda x: x[2], reverse=True)[:MAX_OBJECTS]
        selected_person = sorted(detected_person, key= lambda x: x[0], reverse=True)[:MAX_PERSONS]

        hois = get_ids(selected_objects, hoi_info)

        image_black = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for general_object in detected_objects:
            image_black[round(general_object[1][1]):round(general_object[1][3]), round(general_object[1][0]):round(general_object[1][2])] = 0

        for pers in selected_person:
            results_person = {'image_id': im_id,
                              'im_path': im_path,
                              'person_box': pers[1]}

            results_hoi = [None] * 3
            for hoi in hois:
                if hoi[0][f"{hoi[0]['action']}_agent"] == 0:
                    results_hoi[0] = hoi[0][f"{hoi[0]['action']}_agent"]
                    results_hoi[1] = hoi[0][f"{hoi[0]['action']}_obj"]
                    results_hoi[2] = hoi[0][f"{hoi[0]['action']}_instr"]

                else:
                    results_hoi[0] = 0
                    for interaction in hoi:
                        filepath = osp.join(save_images_path, f"{im_path}_{image_count}.png")
                        obj = interaction[f"{interaction['action']}_obj"]
                        instr = interaction[f"{interaction['action']}_instr"]
                        question = interaction['question']
                        context_objects = None

                        if mode == 'Baseline':
                            filepath = img_filepath
                        
                        elif mode == 'Bboxes':
                            thickness = max(int(min(image.size)/200), 1)
                            image_copy = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)
                            if len(obj) == 4:
                                cv2.rectangle(image_copy, (round(obj[0]), round(obj[1])), (round(obj[2]), round(obj[3])), (0, 0, 255), thickness)
                            if len(instr) == 4:
                                cv2.rectangle(image_copy, (round(instr[0]), round(instr[1])), (round(instr[2]), round(instr[3])), (0, 255, 0), thickness)
                            cv2.rectangle(image_copy, (round(pers[1][0]), round(pers[1][1])), (round(pers[1][2]), round(pers[1][3])), (255, 0, 0), thickness)
                            cv2.imwrite(filepath, image_copy)
                        
                        elif mode == 'Crop':
                            min_x = [round(pers[1][0])]
                            min_y = [round(pers[1][1])]
                            max_x = [round(pers[1][2])]
                            max_y = [round(pers[1][3])]
                            if len(obj) == 4:
                                min_x.append(round(obj[0]))
                                min_y.append(round(obj[1]))
                                max_x.append(round(obj[2]))
                                max_y.append(round(obj[3]))
                            if len(instr) == 4:
                                min_x.append(round(instr[0]))
                                min_y.append(round(instr[1]))
                                max_x.append(round(instr[2]))
                                max_y.append(round(instr[3]))
                            
                            x1, y1 = min(min_x), min(min_y)
                            x2, y2 = max(max_x), max(max_y)
                            x1, x2 = min_factor(x1, x2, 0, image.size[0]-1)
                            y1, y2 = min_factor(y1, y2, 0, image.size[1]-1)
                            image_copy = image.crop((x1, y1, x2, y2))
                            image_copy.save(filepath)
                        
                        elif mode == 'Blur':
                            image_aux = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            image_copy = image_aux.copy()
                            image_copy = cv2.blur(image_copy, ksize)
                            if len(obj) == 4:
                                image_copy[round(obj[1]):round(obj[3]), round(obj[0]):round(obj[2])] = image_aux[round(obj[1]):round(obj[3]), round(obj[0]):round(obj[2])]
                            if len(instr) == 4:
                                image_copy[round(instr[1]):round(instr[3]), round(instr[0]):round(instr[2])] = image_aux[round(instr[1]):round(instr[3]), round(instr[0]):round(instr[2])]
                            image_copy[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])] = image_aux[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])]
                            cv2.imwrite(filepath, image_copy)
                        
                        elif mode == 'Context':
                            context_objects = [image.crop(obj[:4]), image.crop(pers[1])]
                            filepath = img_filepath
                        
                        elif mode == 'Gray':
                            image_aux = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            image_copy = image_aux.copy()
                            image_copy = cv2.cvtColor(cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                            if len(obj) == 4:
                                image_copy[round(obj[1]):round(obj[3]), round(obj[0]):round(obj[2])] = image_aux[round(obj[1]):round(obj[3]), round(obj[0]):round(obj[2])]
                            if len(instr) == 4:
                                image_copy[round(instr[1]):round(instr[3]), round(instr[0]):round(instr[2])] = image_aux[round(instr[1]):round(instr[3]), round(instr[0]):round(instr[2])]
                            image_copy[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])] = image_aux[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])]
                            cv2.imwrite(filepath, image_copy)
                        
                        elif mode == 'Black':
                            image_aux = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            image_copy = image_black.copy()
                            if len(obj) == 4:
                                image_copy[round(obj[1]):round(obj[3]), round(obj[0]):round(obj[2])] = image_aux[round(obj[1]):round(obj[3]), round(obj[0]):round(obj[2])]
                            if len(instr) == 4:
                                image_copy[round(instr[1]):round(instr[3]), round(instr[0]):round(instr[2])] = image_aux[round(instr[1]):round(instr[3]), round(instr[0]):round(instr[2])]
                            image_copy[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])] = image_aux[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])]
                            cv2.imwrite(filepath, image_copy)
                        
                        elif mode == 'Marker':
                            image_copy = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)
                            thickness = max(int(min(image.size)/200), 1)
                            person = (round((pers[1][2]-pers[1][0])/2 + pers[1][0]), round((pers[1][3]-pers[1][1])/2 + pers[1][1]))
                            person_length = (round(math.sqrt(2)*(pers[1][2]-pers[1][0])/2), round(math.sqrt(2)*(pers[1][3]-pers[1][1])/2))
                            cv2.ellipse(image_copy, person, person_length, angle, startAngle, endAngle, (255, 0, 0), thickness)
                                                        
                            if len(obj) == 4:
                                object_item = (round((obj[2]-obj[0])/2 + obj[0]), round((obj[3]-obj[1])/2 + obj[1]))
                                object_length = (round(math.sqrt(2)*(obj[2]-obj[0])/2), round(math.sqrt(2)*(obj[3]-obj[1])/2))
                                cv2.ellipse(image_copy, object_item, object_length, angle, startAngle, endAngle, (0, 0, 255), thickness)
                            if len(instr) == 4:
                                instr_item = (round((instr[2]-instr[0])/2 + instr[0]), round((instr[3]-instr[1])/2 + instr[1]))
                                instr_length = (round(math.sqrt(2)*(instr[2]-instr[0])/2), round(math.sqrt(2)*(instr[3]-instr[1])/2))
                                cv2.ellipse(image_copy, instr_item, instr_length, angle, startAngle, endAngle, (0, 255, 0), thickness)                      

                            cv2.imwrite(filepath, image_copy)
                        
                        elif mode == 'Text_center':
                            person = (round((pers[1][2]-pers[1][0])/2 + pers[1][0]), round((pers[1][3]-pers[1][1])/2 + pers[1][1]))
                            if len(obj) == 4 and len(instr) == 4:
                                object_item = (round((obj[2]-obj[0])/2 + obj[0]), round((obj[3]-obj[1])/2 + obj[1]))
                                instr_item = (round((instr[2]-instr[0])/2 + instr[0]), round((instr[3]-instr[1])/2 + instr[1]))
                                question = f"Is the person centered at {person} {interaction['action_gerund']} the " + \
                                           f"{interaction['obj']} centered at {object_item} with the " + \
                                           f"{interaction['instr']} centered at {instr_item}? Answer with yes or no."
                            elif len(obj) == 4:
                                object_item = (round((obj[2]-obj[0])/2 + obj[0]), round((obj[3]-obj[1])/2 + obj[1]))
                                question = f"Is the person centered at {person} {interaction['action_gerund']} the " + \
                                           f"{interaction['obj']} centered at {object_item}? Answer with yes or no."
                            elif len(instr) == 4:
                                instr_item = (round((instr[2]-instr[0])/2 + instr[0]), round((instr[3]-instr[1])/2 + instr[1]))
                                question = f"Is the person centered at {person} {interaction['action_gerund']} with the " + \
                                           f"{interaction['instr']} centered at {instr_item}? Answer with yes or no."
                            else:
                                question = f"Is the person centered at {person} {interaction['action_gerund']}? Answer with yes or no."
                            filepath = img_filepath
                            
                        elif mode == 'Text_boundary':
                            person = [(round(pers[1][0]), round(pers[1][1])), (round(pers[1][0]), round(pers[1][3])), (round(pers[1][2]), round(pers[1][1])), (round(pers[1][2]), round(pers[1][3]))]
                            person = f"{person[0]}, {person[1]}, {person[2]} and {person[3]}"
                            if len(obj) == 4 and len(instr) == 4:
                                object_item = [(round(obj[0]), round(obj[1])), (round(obj[0]), round(obj[3])), (round(obj[2]), round(obj[1])), (round(obj[2]), round(obj[3]))]
                                object_item = f"{object_item[0]}, {object_item[1]}, {object_item[2]} and {object_item[3]}"
                                instr_item = [(round(instr[0]), round(instr[1])), (round(instr[0]), round(instr[3])), (round(instr[2]), round(instr[1])), (round(instr[2]), round(instr[3]))]
                                instr_item = f"{instr_item[0]}, {instr_item[1]}, {instr_item[2]} and {instr_item[3]}"
                                question = f"Is the person within image coordinates {person} {interaction['action_gerund']} the " + \
                                           f"{interaction['obj']} within image coordinates {object_item} with the " + \
                                           f"{interaction['instr']} within image coordinates {instr_item}? Answer yes or no."
                            elif len(obj) == 4:
                                object_item = [(round(obj[0]), round(obj[1])), (round(obj[0]), round(obj[3])), (round(obj[2]), round(obj[1])), (round(obj[2]), round(obj[3]))]
                                object_item = f"{object_item[0]}, {object_item[1]}, {object_item[2]} and {object_item[3]}"
                                question = f"Is the person within image coordinates {person} {interaction['action_gerund']} the " + \
                                           f"{interaction['obj']} within image coordinates {object_item}? Answer yes or no."
                            elif len(instr) == 4:
                                instr_item = [(round(instr[0]), round(instr[1])), (round(instr[0]), round(instr[3])), (round(instr[2]), round(instr[1])), (round(instr[2]), round(instr[3]))]
                                instr_item = f"{instr_item[0]}, {instr_item[1]}, {instr_item[2]} and {instr_item[3]}"
                                question = f"Is the person within image coordinates {person} {interaction['action_gerund']} with the " + \
                                           f"{interaction['instr']} within image coordinates {instr_item}? Answer yes or no."
                            else:
                                question = f"Is the person within image coordinates {person} {interaction['action_gerund']}? Answer yes or no."                          
                            filepath = img_filepath
                        
                        elif mode == 'Crop_black':
                            min_x = [round(pers[1][0])]
                            min_y = [round(pers[1][1])]
                            max_x = [round(pers[1][2])]
                            max_y = [round(pers[1][3])]
                            if len(obj) == 4:
                                min_x.append(round(obj[0]))
                                min_y.append(round(obj[1]))
                                max_x.append(round(obj[2]))
                                max_y.append(round(obj[3]))
                            if len(instr) == 4:
                                min_x.append(round(instr[0]))
                                min_y.append(round(instr[1]))
                                max_x.append(round(instr[2]))
                                max_y.append(round(instr[3]))
                            
                            x1, y1 = min(min_x), min(min_y)
                            x2, y2 = max(max_x), max(max_y)
                            x1, x2 = min_factor(x1, x2, 0, image.size[0]-1)
                            y1, y2 = min_factor(y1, y2, 0, image.size[1]-1)

                            image_aux = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            image_copy = image_black.copy()
                            if len(obj) == 4:
                                image_copy[round(obj[1]):round(obj[3]), round(obj[0]):round(obj[2])] = image_aux[round(obj[1]):round(obj[3]), round(obj[0]):round(obj[2])]
                            if len(instr) == 4:
                                image_copy[round(instr[1]):round(instr[3]), round(instr[0]):round(instr[2])] = image_aux[round(instr[1]):round(instr[3]), round(instr[0]):round(instr[2])]
                            image_copy[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])] = image_aux[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])]
                            image_copy = image_copy[y1:y2, x1:x2]
                            cv2.imwrite(filepath, image_copy)
                    
                        elif mode == 'Crop_blur':
                            min_x = [round(pers[1][0])]
                            min_y = [round(pers[1][1])]
                            max_x = [round(pers[1][2])]
                            max_y = [round(pers[1][3])]
                            if len(obj) == 4:
                                min_x.append(round(obj[0]))
                                min_y.append(round(obj[1]))
                                max_x.append(round(obj[2]))
                                max_y.append(round(obj[3]))
                            if len(instr) == 4:
                                min_x.append(round(instr[0]))
                                min_y.append(round(instr[1]))
                                max_x.append(round(instr[2]))
                                max_y.append(round(instr[3]))
                            
                            x1, y1 = min(min_x), min(min_y)
                            x2, y2 = max(max_x), max(max_y)
                            x1, x2 = min_factor(x1, x2, 0, image.size[0]-1)
                            y1, y2 = min_factor(y1, y2, 0, image.size[1]-1)

                            image_aux = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            image_copy = image_aux.copy()
                            image_copy = cv2.blur(image_copy, ksize)
                            if len(obj) == 4:
                                image_copy[round(obj[1]):round(obj[3]), round(obj[0]):round(obj[2])] = image_aux[round(obj[1]):round(obj[3]), round(obj[0]):round(obj[2])]
                            if len(instr) == 4:
                                image_copy[round(instr[1]):round(instr[3]), round(instr[0]):round(instr[2])] = image_aux[round(instr[1]):round(instr[3]), round(instr[0]):round(instr[2])]
                            image_copy[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])] = image_aux[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])]
                            image_copy = image_copy[y1:y2, x1:x2]
                            cv2.imwrite(filepath, image_copy)
                        
                        elif mode == 'Crop_context':
                            min_x = [round(pers[1][0])]
                            min_y = [round(pers[1][1])]
                            max_x = [round(pers[1][2])]
                            max_y = [round(pers[1][3])]
                            if len(obj) == 4:
                                min_x.append(round(obj[0]))
                                min_y.append(round(obj[1]))
                                max_x.append(round(obj[2]))
                                max_y.append(round(obj[3]))
                            if len(instr) == 4:
                                min_x.append(round(instr[0]))
                                min_y.append(round(instr[1]))
                                max_x.append(round(instr[2]))
                                max_y.append(round(instr[3]))
                            
                            x1, y1 = min(min_x), min(min_y)
                            x2, y2 = max(max_x), max(max_y)
                            x1, x2 = min_factor(x1, x2, 0, image.size[0]-1)
                            y1, y2 = min_factor(y1, y2, 0, image.size[1]-1)
                            image_copy = image.crop((x1, y1, x2, y2))
                            image_copy.save(filepath)

                            context_objects = [image.crop(obj[:4]), image.crop(pers[1])]

                        model_output, _, score = model.infer(question, filepath, detected_objects=context_objects)

                        if results_hoi[0] < score:
                            results_hoi[0] = score
                            results_hoi[1] = interaction[f"{interaction['action']}_obj"]
                            results_hoi[2] = interaction[f"{interaction['action']}_instr"]
                        
                        image_count += 1
                    
                    if len(results_hoi[1]) == 4:
                        results_hoi[1] = results_hoi[1] + [score]
                    if len(results_hoi[2]) == 4:
                        results_hoi[2] = results_hoi[2] + [score]
                
                if f"{hoi[0]['action']}_agent" not in results_person.keys() or results_person[f"{hoi[0]['action']}_agent"] < results_hoi[0]:
                    results_person[f"{hoi[0]['action']}_agent"] = results_hoi[0]
                    results_person[f"{hoi[0]['action']}_obj"] = results_hoi[1]
                    results_person[f"{hoi[0]['action']}_instr"] = results_hoi[2]
            
            results.append(results_person)
        with open(output_filepath, "wb") as output_file:
            pickle.dump(results, output_file)

    # create parent dir for the file output_filepath
    os.makedirs(osp.dirname(output_filepath), exist_ok=True)  
    with open(output_filepath, "wb") as output_file:
        pickle.dump(results, output_file)


def init_model(model_name) -> BaseModel:
    if "minicpm-v" in model_name.lower():
        return MiniCPM_V(model_name)
    elif "llava-onevision" in model_name.lower():
        return LLaVAOneVisionModel(model_name)
    elif "qwen" in model_name.lower():
        return Qwen2VLModel(model_name)
    elif "internvl2" in model_name.lower():
        return InternVL2Model(model_name)
    elif "phi" in model_name.lower():
        return PhiVisionModel(model_name)
    elif "ovis" in model_name.lower():
        return Ovis2Model(model_name)
#    elif "deepseek-vl2" in model_name.lower():
#        return DeepseekVL2Model(model_name)
    else:
        raise NotImplementedError

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--img-dir", type=str)
    parser.add_argument("--gt-filepath", type=str)
    parser.add_argument("--q-hoi-filepath", type=str)
    parser.add_argument("--output-filepath", type=str)
    parser.add_argument("--mode", type=str, default='Baseline')
    parser.add_argument("--save-folderpath", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    main(args.model_name, args.img_dir, args.gt_filepath, args.q_hoi_filepath, args.output_filepath, args.mode, args.save_folderpath)
