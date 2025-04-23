from vsrl_eval import VCOCOeval

def main(results_filepath):
    vcocoeval = VCOCOeval('./data/vcoco/vcoco_test.json',
                        './data/instances_vcoco_all_2014.json',
                        './data/splits/vcoco_test.ids')
    vcocoeval._do_eval(results_filepath, ovr_thresh=0.5)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-filepath", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    main(args.results_filepath)