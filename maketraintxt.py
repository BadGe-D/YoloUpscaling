import os

# Thư mục chứa ảnh train
image_dir = 'images/train'
# Tên file output
output_file = 'train.txt'

# Duyệt tất cả file trong image_dir và ghi tên file vào train.txt
with open(output_file, 'w') as f:
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            f.write(f'{image_dir}/{filename}\n')

print(f'Đã tạo {output_file} với {image_dir} chứa {len(os.listdir(image_dir))} ảnh.')

