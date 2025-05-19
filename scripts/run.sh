#!/bin/bash

# GAN
python -m src.gan.train_gan \
  --empty_dir data/real \
  --symbols_dir data/synthetic \
  --epochs 100 \
  --batch_size 4 \
  --save_dir models/gan_output

# convert
python -m src.yolo.convert_xml_to_yolo \
  --labels_dir annotations/ \
  --images_dir data/real \
  --class_list downlight

# splits for YOLO
python -m src.yolo.create_splits \
  --real_dir YOLO_data_real \
  --fake_dir YOLO_data_fake \
  --output_root datasets_yolo_split \
  --synthetic_ratios 0 0.25 0.5 0.75 1.0

# train YOLOv5
python -m src.yolo.train_yolo \
  --data_yaml datasets_yolo_split/real_50synth/data.yaml \
  --name real_50synth \
  --project runs_yolo5 \
  --weights yolov5s.pt

# evaluate
python -m src.yolo.evaluate_yolo \
  --ground_truth_dir TEST/ \
  --prediction_dir runs_yolo5/real_50synth/labels \
  --image_size 640