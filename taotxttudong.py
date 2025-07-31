import os

img_dir = '/content/drive/MyDrive/VideoUpScaling/dataset/images/train'  # thư mục chứa ảnh
label_dir = '/content/drive/MyDrive/VideoUpScaling/dataset/labelchuaco'  # thư mục chứa nhãn (YOLO .txt)

os.makedirs(label_dir, exist_ok=True)

for fname in os.listdir(img_dir):
    if fname.endswith(('.jpg', '.png')):
        name = os.path.splitext(fname)[0]
        txt_path = os.path.join(label_dir, name + '.txt')
        if not os.path.exists(txt_path):
            open(txt_path, 'w').close()  # tạo file rỗng
            print(f"Đã tạo file trống cho: {name}")
