import json
import torch
from sklearn.metrics import average_precision_score

def map(res_filepath):
    with open(res_filepath, "r") as f:
        res_dicts = json.load(f)

    results_by_hoi = dict()
    gt_by_hoi = dict()
    for result in res_dicts:
        hoi = result['obj_class'] + '_' + result['action']
        if hoi in results_by_hoi:
            results_by_hoi[hoi] += [result['score']]
        else:
            results_by_hoi[hoi] = [result['score']]
            gt_by_hoi[hoi] = []

        if result['answer'] == 'yes':
            gt_by_hoi[hoi] += [1]
        else:
            gt_by_hoi[hoi] += [0]

    ap_by_hoi = dict()
    for hoi in results_by_hoi.keys():
        ap_by_hoi[hoi] = average_precision_score(gt_by_hoi[hoi], results_by_hoi[hoi])

    map = sum(ap_by_hoi.values())/len(results_by_hoi.keys())
    return map

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-filepath", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    map = map(args.results_filepath)
    print(map)