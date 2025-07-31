import cv2
import os

# === Cấu hình ===
video_path = "datatest/10.72.216.20_.mp4"          # Đường dẫn video gốc
output_dir = "output_frames_tiles"     # Thư mục lưu ảnh
frame_step = 57                   # Mỗi bao nhiêu frame thì lấy (1 = mọi frame)

resize_size = (960, 960)
os.makedirs(output_dir, exist_ok=True)

# === Mở video ===
cap = cv2.VideoCapture(video_path)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_step == 0:
        frame=cv2.resize(frame,resize_size)
        h, w, _ = frame.shape
        h_step = h // 3
        w_step = w // 3

        for i in range(3):
            for j in range(3):
                tile = frame[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                tile_name = f"frame_{frame_id:05d}_tile_{i}_{j}.png"
                tile_path = os.path.join(output_dir, tile_name)
                cv2.imwrite(tile_path, tile)

    frame_id += 1

cap.release()
print("Xong.")
