import os
import yaml
from pathlib import Path

def load_yaml(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Không thể đọc YAML: {e}")
        return None

def check_label_format(file_path):
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    return f"Lỗi dòng {i}: số lượng phần tử != 5"
                class_id, *coords = parts
                if not class_id.isdigit():
                    return f"Lỗi dòng {i}: class_id không phải số nguyên"
                if not all(0 <= float(x) <= 1 for x in coords):
                    return f"Lỗi dòng {i}: tọa độ không nằm trong [0, 1]"
        return None
    except Exception as e:
        return f"Lỗi đọc file: {e}"

def main(dataset_path):
    dataset_path = Path(dataset_path)
    yaml_path = dataset_path / "data.yaml"
    data = load_yaml(yaml_path)
    if data is None:
        return

    names = data.get("names", [])
    if isinstance(names, dict):
        names = list(names.values())
    if not isinstance(names, list):
        print("[ERROR] Trường 'names' phải là list hoặc dict.")
        return
    num_classes = len(names)

    train_txt = dataset_path / data.get("train", "train.txt")
    if not train_txt.exists():
        print(f"[ERROR] Không tìm thấy file train: {train_txt}")
        return

    with open(train_txt, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Tổng số ảnh: {len(image_paths)}")

    for img_rel_path in image_paths:
        img_path = dataset_path / img_rel_path
        if not img_path.exists():
            print(f"[ERROR] Không tìm thấy ảnh: {img_path}")
            continue

        label_rel_path = Path(str(img_rel_path).replace("images", "labels")).with_suffix(".txt")
        label_path = dataset_path / label_rel_path

        if not label_path.exists():
            print(f"[ERROR] Không tìm thấy nhãn: {label_path}")
            continue

        error = check_label_format(label_path)
        if error:
            print(f"[ERROR] {label_path}: {error}")
            continue

        with open(label_path, 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                if class_id >= num_classes:
                    print(f"[ERROR] {label_path}: class_id {class_id} > số lớp ({num_classes - 1})")

    print("[✓] Kiểm tra hoàn tất.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Cách dùng: python check_yolo_dataset.py /path/to/unzipped_dataset_folder")
    else:
        main(sys.argv[1])
