import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
from mmdet.apis import DetInferencer

def coco_classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]

def main(img_dir, qa_filepath, output_filepath, model, confidence=0.1) -> None:
    with open(qa_filepath, "r") as f:
        qa_dicts = json.load(f)

    detector = DetInferencer(model=model, device='cuda')
    results = []
    current_file = ""
    for qa_dict in tqdm(qa_dicts, total=len(qa_dicts)):
        img_filepath = osp.join(img_dir, qa_dict["im_path"])

        if current_file != img_filepath:
            current_file = img_filepath

            detections = detector(img_filepath, pred_score_thr=0.0, draw_pred=False, no_save_vis=True)['predictions'][0]

            detected = {coco_classes()[detections['labels'][i]] for i in range(len(detections['labels'])) if detections['scores'][i] >= confidence}
            scores = dict()
            for i in range(len(detections['labels'])):
                obj_name = coco_classes()[detections['labels'][i]]
                if obj_name not in scores.keys():
                    scores[obj_name] = [detections['scores'][i]]
                else:
                    scores[obj_name] += [detections['scores'][i]]
    
            for name in coco_classes():
                if name in scores.keys():
                    scores[name] = max(scores[name])
                else:
                    scores[name] = 0

            det_scores = {k: scores[k] for k in scores.keys()}

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
    parser.add_argument("--model", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.img_dir, args.qa_filepath, args.output_filepath, args.model)
