from segment_anything import SamPredictor, sam_model_registry

import argparse
import json
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
import numpy as np
from typing import Any, Dict, List
import pyvips

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--level",
    type=int,
    required=True,
    help="Read a specific level in TIF image with pyvips",
)

parser.add_argument(
    "--scale",
    type=float,
    required=True,
    help="The scale of bounding box",
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

object_detection = parser.add_argument_group("object detection")

object_detection.add_argument(
    "--xcenter",
    type=float,
    required=True,
    help="The x center of the bounding box",
)

object_detection.add_argument(
    "--ycenter",
    type=float,
    required=True,
    help="The y center of the bounding box",
)

object_detection.add_argument(
    "--width",
    type=float,
    required=True,
    help="The width of the bounding box",
)

object_detection.add_argument(
    "--height",
    type=float,
    required=True,
    help="The height the bounding box",
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    for i, mask_data in enumerate(masks):
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask_data * 255)
    return mask_data.astype("uint8")


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    mask_predictor = SamPredictor(sam)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        image = pyvips.Image.new_from_file(t, level=args.level)
        image = image.colourspace("srgb")
        image = image.numpy()
        x_center = int(image.shape[1]*args.xcenter)
        y_center = int(image.shape[0]*args.ycenter)
        width = int(image.shape[1]*args.width/2*args.scale)
        height = int(image.shape[0]*args.height/2*args.scale)
        left = max(0, x_center-width)
        right = min(image.shape[1], x_center+width)
        bottom = max(0, y_center-height)
        top = min(image.shape[0], y_center+height)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = image[bottom:top, left:right]
        # ---
        print(image.shape[1], image.shape[0])
        print(x_center, y_center)
        input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
        input_label = np.array([1])
        mask_predictor.set_image(image)
        masks, scores, logits = mask_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        masks, scores, logits = mask_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False
        )
        # ---
        base = os.path.basename(t)
        base = f"{os.path.splitext(base)[0]}_{x_center}_{y_center}"
        save_base = os.path.join(args.output, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=False)
            mask = write_masks_to_folder(masks, save_base)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)

    single_tissue_region = cv2.bitwise_and(image, image, mask=mask)
    vips_img = pyvips.Image.new_from_array(single_tissue_region)
    save_path = f"{save_base}/{base}.svs"
    vips_img.tiffsave(save_path, compression='jpeg', Q='90', tile=True, bigtiff=True,
                      pyramid=True, miniswhite=False, squash=False)
    print('Saved: ', save_path)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)