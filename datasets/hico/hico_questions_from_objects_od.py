import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def main(img_dir, qa_filepath, output_filepath, confidence=0.1) -> None:
    with open(qa_filepath, "r") as f:
        qa_dicts = json.load(f)

    detector = YOLO("yolo11x.pt").to('cuda')
    results = []
    current_file = ""
    for qa_dict in tqdm(qa_dicts, total=len(qa_dicts)):
        img_filepath = osp.join(img_dir, qa_dict["im_path"])

        if current_file != img_filepath:
            current_file = img_filepath

            detections = detector(img_filepath, conf=0.0, verbose=False)[0]
            detected = {detections.names[obj[5].item()].replace(" ", "_") for obj in detections.boxes.data if obj[4] >= confidence}
            scores = dict()
            for obj in detections.boxes.data:
                obj_name = detections.names[obj[5].item()].replace(" ", "_")
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


        if qa_dict['obj_class'] in detected:
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
                    "score": det_scores[qa_dict["obj_class"]]
                })

    # create parent dir for the file output_filepath
    os.makedirs(osp.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w") as f:
        json.dump(results, f, indent=4)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str)
    parser.add_argument("--qa-filepath", type=str)
    parser.add_argument("--output-filepath", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.img_dir, args.qa_filepath, args.output_filepath)
