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

from ultralytics import YOLO
from PIL import Image

def main(model_name, img_dir, qa_filepath, output_filepath) -> None:
    with open(qa_filepath, "r") as f:
        qa_dicts = json.load(f)

    obj_detector = YOLO("yolo11x.pt").to('cuda')
    model = init_model(model_name)
    current_file = ""
    results = []
    for qa_dict in tqdm(qa_dicts, total=len(qa_dicts)):
        img_filepath = osp.join(img_dir, qa_dict["im_path"])

        if current_file != img_filepath:
            cumulative_score = 1
            current_file = img_filepath
            detected_boxes, scores_det = detect_objects(obj_detector, img_filepath)
            detected = {det[0] for det in detected_boxes}

            image = Image.open(img_filepath).convert("RGB")

            for i, obj in enumerate(detected_boxes):
                x1, y1, x2, y2 = obj[1][0], obj[1][1], obj[1][2], obj[1][3]
                detected_boxes[i] = (detected_boxes[i][0], (x1.item(), y1.item(), x2.item(), y2.item()), detected_boxes[i][2])

            bboxes_pers = [(det[1], (det[1][2]-det[1][0])*(det[1][3]-det[1][1])) for det in detected_boxes if det[0] == 'person']
            bboxes_pers.sort(reverse=True, key= lambda x: x[1])
            bboxes_pers = [det[0] for det in bboxes_pers][:MAX_PERSONS]


        if qa_dict['obj_class'] in detected:          
            if qa_dict['action'] != 'no_interaction':
                bboxes_obj = [(det[1], det[2]) for det in detected_boxes if det[0] == qa_dict['obj_class']]
                bboxes_obj.sort(reverse=True, key= lambda x: x[1])
                bboxes_obj = [det[0] for det in bboxes_obj][:MAX_OBJECTS]
                bboxes = bboxes_obj + bboxes_pers
                
                objects = [image.crop(bbox) for bbox in bboxes]

                scores = []
                for prompt in qa_dict["text_prompt"]:
                    model_output, _, score = model.infer(prompt,
                                                            img_filepath,
                                                            detected_objects=objects
                                                            )
                    scores += [score]
                
                score = sum(scores)/len(scores)
                cumulative_score *= 1-score

                results.append(
                    {
                        "im_path": qa_dict["im_path"],
                        "text_prompt": qa_dict["text_prompt"],
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
                    "score": scores_det[qa_dict["obj_class"]]
                })

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
    else:
        raise NotImplementedError

def detect_objects(detector, img_filepath, confidence = 0.1):
    detections = detector(img_filepath, conf=0.0, verbose=False)[0]
    detected = [(detections.names[obj[5].item()].replace(" ", "_"), obj[:4], obj[4]) for obj in detections.boxes.data if obj[4] >= confidence]
    scores = dict()
    for obj in detections.boxes.data:
        obj_name = detections.names[obj[5].item()]
        if obj_name not in scores.keys():
            scores[obj_name] = [obj[4]]
        else:
            scores[obj_name] += [obj[4]]
    
    for name in detections.names.values():
        if name in scores.keys():
            scores[name] = max(scores[name]).item()
        else:
            scores[name] = 0

    det_scores = {k.replace(" ", "_"): scores[k] for k in scores.keys()}
    
    return detected, det_scores

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
