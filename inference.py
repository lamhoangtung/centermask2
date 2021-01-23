import glob
import os
import random

import cv2
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from detectron2.utils.visualizer import ColorMode, Visualizer
# from detectron2.data.catalog import MetadataCatalog
from tqdm import tqdm

from centermask.config import get_cfg

INPUT_TEST_DATA_PATH = 'datasets/publaynet/test'
TEST_AMMOUNT = 5
THRESH_TEST = 0.5
WEIGHT_PATH = 'output/PubLayNet-CenterMask-Lite-V-19-ms-4x/model_0016999.pth'
CONFIG_FILE_PATH = 'output/PubLayNet-CenterMask-Lite-V-19-ms-4x/config.yaml'
OUTPUT_TEST_RESULT_PATH = os.path.join(os.path.dirname(WEIGHT_PATH), 'test_inference_results')
os.makedirs(OUTPUT_TEST_RESULT_PATH, exist_ok=True)
OUTPUT_TEST_RESULT_PATH = os.path.join(OUTPUT_TEST_RESULT_PATH, os.path.basename(WEIGHT_PATH))
os.makedirs(OUTPUT_TEST_RESULT_PATH, exist_ok=True)
print("Saving inference results to:", OUTPUT_TEST_RESULT_PATH)

publaynet_metadata = {'name': 'publaynet_val', 'json_file': '/home/techainer/linus/publaynet/val.json', 'image_root': '/home/techainer/linus/publaynet/val',
                      'evaluator_type': 'coco', 'thing_classes': ['text', 'title', 'list', 'table', 'figure'], 'thing_dataset_id_to_contiguous_id': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}}

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE_PATH)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = WEIGHT_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESH_TEST
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = THRESH_TEST
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = THRESH_TEST
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = THRESH_TEST
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # create a predictor
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)

    all_samples = glob.glob(os.path.join(INPUT_TEST_DATA_PATH, '*'))
    if TEST_AMMOUNT != -1:
        all_samples = random.sample(all_samples, TEST_AMMOUNT)

    for sample in tqdm(all_samples):
        image = cv2.imread(sample)
        outputs = predictor(image)
        outputs = outputs["instances"].to("cpu")
        viz = Visualizer(
            image[:, :, ::-1],
            metadata=publaynet_metadata,
            scale=1,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )
        viz = viz.draw_instance_predictions(outputs)
        visualized_image = viz.get_image()[:, :, ::-1]
        image_path = os.path.join(OUTPUT_TEST_RESULT_PATH, os.path.basename(sample))
        cv2.imwrite(image_path, visualized_image)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )
