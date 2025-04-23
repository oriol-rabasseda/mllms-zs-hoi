import json
import torch
from sklearn.metrics import average_precision_score
import numpy as np
from tqdm import tqdm
from bbox_utils import compute_iou

rare_hoi = [77, 84, 169, 532, 78, 329, 587, 596, 215, 51, 408, 239, 137, 101, 182, 173, 189, 108, 400, 207, 550, 199, 551, 557, 486, 398, 402, 411, 365, 100, 382, 150, 166, 185, 437, 475, 193, 196, 28, 326, 23, 593, 582, 316, 255, 392, 483, 136, 521, 441, 318, 128, 597, 581, 552, 63, 536, 405, 256, 427, 452, 464, 470, 56, 105, 258, 9, 67, 391, 355, 527, 450, 280, 293, 440, 180, 548, 600, 432, 262, 518, 304, 417, 190, 206, 159, 81, 290, 261, 598, 275, 287, 230, 430, 404, 505, 500, 167, 359, 281, 312, 499, 45, 549, 540, 334, 399, 579, 223, 380, 64, 217, 282, 113, 547, 403, 515, 352, 510, 91, 240, 419, 428, 85, 228, 263, 335, 346, 351, 390, 406, 523, 553, 556, 561, 594, 71, 396]

with open('../../../datasets/hico_det/hoi_info.json', "r") as f:
    hoi_info = json.load(f)

def get_ids(hoi_id):
    return [k for k, v in hoi_info.items() if v[0]['obj_class'] == hoi_info[f'{hoi_id:03d}'][0]['obj_class']]

def match_hoi(pred_det,gt_dets):
    is_match = False
    remaining_gt_dets = [gt_det for gt_det in gt_dets]
    for i,gt_det in enumerate(gt_dets):
        human_iou = compute_iou(pred_det['human_box'],gt_det['human_box'])
        if human_iou > 0.5:
            object_iou = compute_iou(pred_det['object_box'],gt_det['object_box'])
            if object_iou > 0.5:
                is_match = True
                del remaining_gt_dets[i]
                break

    return is_match, remaining_gt_dets


def compute_ap(precision,recall):
    if np.any(np.isnan(recall)):
        return np.nan

    ap = 0
    for t in np.arange(0, 1.1, 0.1): # 0, 0.1, 0.2, ..., 1.0
        selected_p = precision[recall>=t]
        if selected_p.size==0:
            p = 0
        else:
            p = np.max(selected_p)   
        ap += p/11.
    
    return ap


def compute_pr(y_true,y_score,npos):
    sorted_y_true = [y for y,_ in 
        sorted(zip(y_true,y_score),key=lambda x: x[1],reverse=True)]
    tp = np.array(sorted_y_true)
    try:
        fp = ~tp
    except:
        precision = np.array([0])
        recall = np.array([0])
        return precision, recall

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos==0:
        recall = np.nan*tp
    else:
        recall = tp / npos
    precision = tp / (tp + fp)
    return precision, recall


def eval_hoi(hoi_id, global_ids, gt_dets, pred_dets):
    y_true = []
    y_score = []
    det_id = []
    npos = 0
    obj_hois = get_ids(hoi_id)
    for global_id in global_ids:
        if int(hoi_id) in gt_dets[global_id].keys():
            candidate_gt_dets = gt_dets[global_id][int(hoi_id)]
        else:
            candidate_gt_dets = []
            '''
            obj_found = False
            for hoi_obj in obj_hois:
                if int(hoi_obj) in gt_dets[global_id].keys():
                    obj_found = True
                    break
            if not obj_found:
                continue
            '''
                
        npos += len(candidate_gt_dets)

        if global_id in pred_dets.keys() and int(hoi_id) in pred_dets[global_id].keys():
            pred_hois = pred_dets[global_id][int(hoi_id)]
        else:
            pred_hois = []

        num_dets = len(pred_hois)
        scores = [hoi['score'] for hoi in pred_hois]
        sorted_idx = [idx for idx, _ in sorted(zip(range(num_dets), scores),
                                               key=lambda x: x[1],
                                               reverse=True)
                     ]

        for i in sorted_idx:
            pred_det = pred_hois[i]
            is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id, i))

    precision,recall = compute_pr(y_true,y_score,npos)
    
    # Compute AP
    ap = compute_ap(precision,recall)

    return (ap, hoi_id)


def load_dets(det_dict, global_ids_set):
    dets = {}
    for im_id in global_ids_set:
        dets[im_id] = {}
        if im_id in det_dict.keys():
            for hoi in det_dict[im_id]:
                if hoi['hoi_id'] not in dets[im_id].keys():
                    dets[im_id][hoi['hoi_id']] = []
        
                det = {
                    'human_box': hoi['sub_bbox'],
                    'object_box': hoi['obj_bbox'],
                }

                if 'score' in hoi.keys():
                    det['score'] = hoi['score']
                dets[im_id][hoi['hoi_id']].append(det)

    return dets


def main(gt_dicts, pred_dicts):    
    global_ids_set = gt_dicts.keys()

    gt_dets = load_dets(gt_dicts, global_ids_set)    
    pred_dets = load_dets(pred_dicts, global_ids_set)

    mAP = []
    for i in tqdm(range(1, 601), total=600):
        mAP.append(eval_hoi(i, global_ids_set, gt_dets, pred_dets))

    count, map_ = 0, 0
    count_rare, map_rare = 0, 0
    count_nonrare, map_nonrare = 0, 0
    for ap, hoi_id in mAP:
        if not np.isnan(ap):
            count += 1
            map_ += ap

            if hoi_id in rare_hoi:
                count_rare += 1
                map_rare += ap
            else:
                count_nonrare += 1
                map_nonrare += ap
    
    print(f"Rare: {map_rare/count_rare}\tNon-rare: {map_nonrare/count_nonrare}")

    return map_/count, mAP

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-filepath", type=str)
    parser.add_argument("--gt-filepath", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.gt_filepath, "r") as f:
        gt_dicts = json.load(f)
    
    with open(args.results_filepath, "r") as f:
        pred_dicts = json.load(f)

    map_res, _ = main(gt_dicts, pred_dicts)
    print(f"Full: {map_res}")