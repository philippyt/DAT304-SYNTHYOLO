import os
import shutil
import random
from pathlib import Path

def create_yolo_splits(real_dir, fake_dir, output_root, synthetic_ratios):
    def get_pairs(folder):
        images = list(Path(folder).glob("*.png"))
        return [(img, img.with_suffix(".xml")) for img in images if img.with_suffix(".xml").exists()]

    real_pairs = get_pairs(real_dir)
    fake_pairs = get_pairs(fake_dir)

    for ratio in synthetic_ratios:
        synth_count = min(int(len(real_pairs) * ratio), len(fake_pairs))
        selected_fake = random.sample(fake_pairs, synth_count) if synth_count > 0 else []
        combined_pairs = real_pairs + selected_fake

        folder_name = f"real_{int(ratio*100)}synth" if ratio > 0 else "real_only"
        images_out = Path(output_root) / folder_name / "images"
        labels_out = Path(output_root) / folder_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for img_path, xml_path in combined_pairs:
            shutil.copy(img_path, images_out / img_path.name)
            shutil.copy(xml_path, labels_out / xml_path.name)