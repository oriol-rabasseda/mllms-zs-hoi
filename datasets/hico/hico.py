import scipy.io
import os.path as osp
import numpy as np
import json
from tqdm import tqdm

def main(mat_file_path: str, output_filepath: str):
    """Convert .mat file to .json file
    """
    # Load the mat file into a dictionary
    data = scipy.io.loadmat(mat_file_path)

    # Access the array or nd-array from the dictionary
    train_data = data["list_test"] # train_data shape is (num_images, 1)

    count_uncertain, count_positive, count_negative = 0, 0, 0

    new_hoi_list = []
    for i, image_item in tqdm(enumerate(train_data), total=len(train_data)):
        img_filename = image_item[0].item()
        annotated = [(data['list_action'][j], label) for j, label in enumerate(data['anno_test'][:,i])]
        new_hoi_one_image = []

        for element in annotated:
            new_hoi = {
                "im_path": img_filename,
                "obj_class": element[0][0][0].item(),
                "action": element[0][0][1].item(),
                "action_gerund": element[0][0][2].item(),
                "label": element[1]
            }

            # Add a hoi for this image
            new_hoi_one_image.append(new_hoi)

            if element[1] == 1:
                count_positive += 1
            elif element[1] == 0:
                count_uncertain += 1
            elif element[1] == -1:
                count_negative += 1

        # Add all hois for this image
        new_hoi_list.append(new_hoi_one_image)

    # Save new_hoi_list a JSON file
    with open(output_filepath, "w") as new_jsonl_file:
        json.dump(new_hoi_list, new_jsonl_file, indent=4)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hico-ann-filepath", type=str)
    parser.add_argument("--output-filepath", type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.hico_ann_filepath, args.output_filepath)