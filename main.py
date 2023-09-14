#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import subprocess
import csv

def main():
    subprocess.run(['python', './tissue_extractor/prompt_generator/infer.py',
                    '--yaml', './tissue_extractor/prompt_generator/yolov6/dataset.yaml',
                    '--weights', './tissue_extractor/prompt_generator/yolov6/weight/best_ckpt.pt',
                    '--source', '/work/u4307600/nlac_playground/lymphocyte/109-P08_HF2_L+I.svs',
                    '--save-dir', 'prompt_results',
                    '--device', '0', '--save-txt']) # , '--not-save-img'

    with open('./prompt_results/109-P08_HF2_L+I.txt', 'r') as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=' ')
        bounding_box = list(csv_reader)

    for cnt in range(len(bounding_box)):
        print(cnt)
        run_sam(bounding_box, cnt)


def run_sam(bounding_box, cnt):
    subprocess.run(['python', './tissue_extractor/segment_anything/amg_wsi.py',
                    '--checkpoint', './tissue_extractor/segment_anything/weight/sam_vit_h_4b8939.pth',
                    '--model-type', 'vit_h',
                    '--input', '/work/u4307600/nlac_playground/lymphocyte/109-P08_HF2_L+I.svs',
                    '--level', '2',
                    '--scale', '1.1',
                    '--output', 'out4',
                    '--xcenter', bounding_box[cnt][1],
                    '--ycenter', bounding_box[cnt][2],
                    '--width', bounding_box[cnt][3],
                    '--height', bounding_box[cnt][4],
                    ])


if __name__ == "__main__":
    main()
