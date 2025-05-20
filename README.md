# Downlight Detection in Floor Plans Using YOLO and GAN-Generated Datasets (DAT304)

By Philip André Haugen and Fredrik Noddeland.

This project uses a GAN to generate synthetic floor plans and evaluates YOLO models trained with varying real/synthetic data ratios.

Note: lack of images in 'data' folder due to GDPR.

## Structure
- `src/gan`: GAN models and training
- `src/yolo`: YOLO training/evaluation, data formatting
- `src/utils`: Shared functions
- `data/`, `datasets_yolo_split/`: Data

## Setup
```bash
pip install -r requirements.txt
```

## Full Pipeline
```bash
bash scripts/run.sh
```
