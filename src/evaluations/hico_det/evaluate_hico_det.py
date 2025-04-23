import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import cv2
import math

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

# ksize 
ksize = (20, 20)

angle = 0
startAngle = 0
endAngle = 360

def get_ids(obj, hoi_info):
    return [hoi for hoi in hoi_info.keys() if hoi_info[hoi]['obj_class'] == obj]

def min_factor(x1, x2, min_num, max_num):
    if x2 - x1 < MIN_FACTOR:
        aug = int((MIN_FACTOR - x2 + x1 + 1)/2)
        x1 -= aug
        if x1 < min_num:
            supp2 = min_num - x1
            x1 = min_num
        else:
            supp2 = 0
        
        x2 += aug
        if x2 > max_num:
            supp1 = max_num - x2
            x2 = max_num
        else:
            supp1 = 0
    
        return x1 + supp1, x2 + supp2

    else:
        return x1, x2


def main(model_name, img_dir, global_ids, qa_filepath, output_filepath, mode, save_images_path):
    with open(qa_filepath, "r") as f:
        qa_dicts = json.load(f)
    
    with open('../../../datasets/hico_det/hoi_info.json', "r") as f:
        hoi_info = json.load(f)
    
    obj_detector = YOLO("yolo11x.pt").to('cuda')
    model = init_model(model_name)
    results = dict()
    current_file = ""
    for im_id in tqdm(global_ids, total=len(global_ids)):
        img_filepath = osp.join(img_dir, im_id + '.jpg')
        image = Image.open(img_filepath).convert("RGB")
        detections = obj_detector(image, conf=0.5, verbose=False)[0]
        cumulative_score = 1
        im_results = []
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

        image_black = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for general_object in detected_objects:
            image_black[round(general_object[1][1]):round(general_object[1][3]), round(general_object[1][0]):round(general_object[1][2])] = 0

        for obj in selected_objects:
            new_obj = obj[0].replace(' ', '_')
            ids = get_ids(new_obj, qa_dicts)
            detected_objects = None

            for pers in selected_person:
                if obj[1] == pers[1]:
                    continue
                
                filepath = osp.join(save_images_path, f"{im_id}_{image_count}.png")

                if mode == 'Baseline':
                    filepath = img_filepath
                
                elif mode == 'Bboxes':
                    thickness = max(int(min(image.size)/200), 1)
                    image_copy = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)
                    cv2.rectangle(image_copy, (round(obj[1][0]), round(obj[1][1])), (round(obj[1][2]), round(obj[1][3])), (0, 0, 255), thickness)
                    cv2.rectangle(image_copy, (round(pers[1][0]), round(pers[1][1])), (round(pers[1][2]), round(pers[1][3])), (255, 0, 0), thickness)
                    cv2.imwrite(filepath, image_copy)
                
                elif mode == 'Crop':
                    x1, y1 = min(round(obj[1][0]), round(pers[1][0])), min(round(obj[1][1]), round(pers[1][1]))
                    x2, y2 = max(round(obj[1][2]), round(pers[1][2])), max(round(obj[1][3]), round(pers[1][3]))
                    x1, x2 = min_factor(x1, x2, 0, image.size[0]-1)
                    y1, y2 = min_factor(y1, y2, 0, image.size[1]-1)
                    image_copy = image.crop((x1, y1, x2, y2))
                    image_copy.save(filepath)

                elif mode == 'Blur':
                    image_aux = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    image_copy = image_aux.copy()
                    image_copy = cv2.blur(image_copy, ksize)
                    image_copy[round(obj[1][1]):round(obj[1][3]), round(obj[1][0]):round(obj[1][2])] = image_aux[round(obj[1][1]):round(obj[1][3]), round(obj[1][0]):round(obj[1][2])]
                    image_copy[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])] = image_aux[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])]
                    cv2.imwrite(filepath, image_copy)
                
                elif mode == 'Context':
                    detected_objects = [image.crop(obj[1]), image.crop(pers[1])]
                    filepath = img_filepath
                
                elif mode == 'Gray':
                    image_aux = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    image_copy = image_aux.copy()
                    image_copy = cv2.cvtColor(cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                    image_copy[round(obj[1][1]):round(obj[1][3]), round(obj[1][0]):round(obj[1][2])] = image_aux[round(obj[1][1]):round(obj[1][3]), round(obj[1][0]):round(obj[1][2])]
                    image_copy[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])] = image_aux[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])]
                    cv2.imwrite(filepath, image_copy)
                
                elif mode == 'Black':
                    image_aux = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    image_copy = image_black.copy()
                    image_copy[round(obj[1][1]):round(obj[1][3]), round(obj[1][0]):round(obj[1][2])] = image_aux[round(obj[1][1]):round(obj[1][3]), round(obj[1][0]):round(obj[1][2])]
                    image_copy[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])] = image_aux[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])]
                    cv2.imwrite(filepath, image_copy)
                
                elif mode == 'Marker':
                    thickness = max(int(min(image.size)/200), 1)
                    person = (round((pers[1][2]-pers[1][0])/2 + pers[1][0]), round((pers[1][3]-pers[1][1])/2 + pers[1][1]))
                    person_length = (round(math.sqrt(2)*(pers[1][2]-pers[1][0])/2), round(math.sqrt(2)*(pers[1][3]-pers[1][1])/2))
                    
                    object_item = (round((obj[1][2]-obj[1][0])/2 + obj[1][0]), round((obj[1][3]-obj[1][1])/2 + obj[1][1]))
                    object_length = (round(math.sqrt(2)*(obj[1][2]-obj[1][0])/2), round(math.sqrt(2)*(obj[1][3]-obj[1][1])/2))

                    image_copy = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)
                    cv2.ellipse(image_copy, object_item, object_length, angle, startAngle, endAngle, (0, 0, 255), thickness)
                    cv2.ellipse(image_copy, person, person_length, angle, startAngle, endAngle, (255, 0, 0), thickness)
                    cv2.imwrite(filepath, image_copy)
                
                elif mode == 'Text_center':
                    person = (round((pers[1][2]-pers[1][0])/2 + pers[1][0]), round((pers[1][3]-pers[1][1])/2 + pers[1][1]))
                    object_item = (round((obj[1][2]-obj[1][0])/2 + obj[1][0]), round((obj[1][3]-obj[1][1])/2 + obj[1][1]))
                    filepath = img_filepath
                
                elif mode == 'Text_boundary':
                    person = [(round(pers[1][0]), round(pers[1][1])), (round(pers[1][0]), round(pers[1][3])), (round(pers[1][2]), round(pers[1][1])), (round(pers[1][2]), round(pers[1][3]))]
                    object_item = [(round(obj[1][0]), round(obj[1][1])), (round(obj[1][0]), round(obj[1][3])), (round(obj[1][2]), round(obj[1][1])), (round(obj[1][2]), round(obj[1][3]))]
                    person = f"{person[0]}, {person[1]}, {person[2]} and {person[3]}"
                    object_item = f"{object_item[0]}, {object_item[1]}, {object_item[2]} and {object_item[3]}"
                    filepath = img_filepath
                
                elif mode == 'Crop_black':
                    x1, y1 = min(round(obj[1][0]), round(pers[1][0])), min(round(obj[1][1]), round(pers[1][1]))
                    x2, y2 = max(round(obj[1][2]), round(pers[1][2])), max(round(obj[1][3]), round(pers[1][3]))
                    x1, x2 = min_factor(x1, x2, 0, image.size[0]-1)
                    y1, y2 = min_factor(y1, y2, 0, image.size[1]-1)

                    image_aux = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    image_copy = image_black.copy()
                    image_copy[round(obj[1][1]):round(obj[1][3]), round(obj[1][0]):round(obj[1][2])] = image_aux[round(obj[1][1]):round(obj[1][3]), round(obj[1][0]):round(obj[1][2])]
                    image_copy[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])] = image_aux[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])]
                    image_copy = image_copy[y1:y2, x1:x2]
                    cv2.imwrite(filepath, image_copy)
            
                elif mode == 'Crop_blur':
                    x1, y1 = min(round(obj[1][0]), round(pers[1][0])), min(round(obj[1][1]), round(pers[1][1]))
                    x2, y2 = max(round(obj[1][2]), round(pers[1][2])), max(round(obj[1][3]), round(pers[1][3]))
                    x1, x2 = min_factor(x1, x2, 0, image.size[0]-1)
                    y1, y2 = min_factor(y1, y2, 0, image.size[1]-1)

                    image_aux = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    image_copy = image_aux.copy()
                    image_copy = cv2.blur(image_copy, ksize)
                    image_copy[round(obj[1][1]):round(obj[1][3]), round(obj[1][0]):round(obj[1][2])] = image_aux[round(obj[1][1]):round(obj[1][3]), round(obj[1][0]):round(obj[1][2])]
                    image_copy[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])] = image_aux[round(pers[1][1]):round(pers[1][3]), round(pers[1][0]):round(pers[1][2])]
                    image_copy = image_copy[y1:y2, x1:x2]
                    cv2.imwrite(filepath, image_copy)
                
                elif mode == 'Crop_context':
                    x1, y1 = min(round(obj[1][0]), round(pers[1][0])), min(round(obj[1][1]), round(pers[1][1]))
                    x2, y2 = max(round(obj[1][2]), round(pers[1][2])), max(round(obj[1][3]), round(pers[1][3]))
                    x1, x2 = min_factor(x1, x2, 0, image.size[0]-1)
                    y1, y2 = min_factor(y1, y2, 0, image.size[1]-1)
                    image_copy = image.crop((x1, y1, x2, y2))
                    image_copy.save(filepath)

                    detected_objects = [image.crop(obj[1]), image.crop(pers[1])]

                for hoi_id in ids:
                    action = qa_dicts[hoi_id]['action']

                    if mode == 'Text_center':
                        object_text = hoi_info[hoi_id][0]['obj_class'].replace('_', ' ')
                        action_text = hoi_info[hoi_id][0]['action_gerund'].replace('_', ' ')

                        question = f"Is the person centered at {person} {action_text} the " + \
                               f"{object_text} centered at {object_item}? Answer yes or no."
                    
                    elif mode == 'Text_boundary':
                        object_text = hoi_info[hoi_id][0]['obj_class'].replace('_', ' ')
                        action_text = hoi_info[hoi_id][0]['action_gerund'].replace('_', ' ')

                        question = f"Is the person within image coordinates {person} {action_text} the " + \
                                f"{object_text} within image coordinates {object_item}? Answer yes or no."
                    
                    else:
                        question = qa_dicts[hoi_id]['questions'][0]
                    
                    if action != 'no_interaction':
                        model_output, _, score = model.infer(question, filepath, detected_objects=detected_objects)
                        cumulative_score *= 1-score

                        im_results.append(
                            {
                                "text_prompt": question,
                                "obj_class": new_obj,
                                "action": action,
                                "model_output": model_output,
                                "score": score,
                                "hoi_id": int(hoi_id),
                                "sub_bbox": pers[1],
                                "obj_bbox": obj[1]
                            })
                        
                    else:
                        im_results.append(
                            {
                                "text_prompt": "",
                                "obj_class": new_obj,
                                "action": action,
                                "model_output": "",
                                "score": cumulative_score,
                                "hoi_id": int(hoi_id),
                                "sub_bbox": pers[1],
                                "obj_bbox": obj[1]
                            })
                        
                        cumulative_score = 1

                image_count += 1
        
        results[im_id] = im_results

    # create parent dir for the file output_filepath
    os.makedirs(osp.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w") as f:
        json.dump(results, f, indent=4)


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

    with open(args.gt_filepath, "r") as f:
        gt_dicts = json.load(f)
    
    global_ids = list(gt_dicts.keys())

    main(args.model_name, args.img_dir, global_ids, args.q_hoi_filepath, args.output_filepath, args.mode, args.save_folderpath)
