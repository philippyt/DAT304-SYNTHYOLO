import xml.etree.ElementTree as ET
from pathlib import Path

def convert_annotations(labels_dir, images_dir, class_list):
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    for xml_file in labels_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_path = images_dir / root.find("filename").text
        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_list:
                continue
            class_id = class_list.index(class_name)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            x_center = (xmin + xmax) / 2.0 / image_width
            y_center = (ymin + ymax) / 2.0 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        txt_file = labels_dir / f"{xml_file.stem}.txt"
        with open(txt_file, "w") as f:
            f.write("\n".join(yolo_lines))