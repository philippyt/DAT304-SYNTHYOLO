import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def yolo_to_xyxy(box, img_w, img_h):
    x_c, y_c, w, h = box
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return [x1, y1, x2, y2]

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def evaluate_model(ground_truth_dir, prediction_dir, image_size=640):
    ious = []
    for filename in tqdm(os.listdir(ground_truth_dir)):
        if not filename.endswith(".txt"):
            continue
        gt_path = os.path.join(ground_truth_dir, filename)
        pred_path = os.path.join(prediction_dir, filename)
        if not os.path.exists(pred_path):
            continue

        gt_boxes = np.loadtxt(gt_path).reshape(-1, 5)[:, 1:]
        pred_boxes = np.loadtxt(pred_path).reshape(-1, 5)[:, 1:]

        gt_boxes = [yolo_to_xyxy(b, image_size, image_size) for b in gt_boxes]
        pred_boxes = [yolo_to_xyxy(b, image_size, image_size) for b in pred_boxes]

        for pred_box in pred_boxes:
            best_iou = 0
            for gt_box in gt_boxes:
                best_iou = max(best_iou, compute_iou(pred_box, gt_box))
            ious.append(best_iou)

    if ious:
        ious = np.array(ious)
        return {
            "mean": ious.mean(),
            "median": np.median(ious),
            "min": ious.min(),
            "max": ious.max(),
            "std": ious.std(),
        }
    return {}