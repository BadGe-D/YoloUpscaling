import cv2
import os

input_folder = "output_frames_tiles"
output_folder = "reconstructed_frames"
os.makedirs(output_folder, exist_ok=True)

# Danh sách các frame id bạn đã xử lý
frame_ids = sorted(set([
    int(fname.split("_")[1]) for fname in os.listdir(input_folder)
    if fname.endswith(".png")
]))

for frame_id in frame_ids:
    try:
        tiles = []
        for i in range(3):
            row = []
            for j in range(3):
                tile_name = f"frame_{frame_id:05d}_tile_{i}_{j}.png"
                tile_path = os.path.join(input_folder, tile_name)
                img = cv2.imread(tile_path)
                if img is None:
                    raise FileNotFoundError(f"Không tìm thấy {tile_path}")
                row.append(img)
            tiles.append(row)

        rows = [cv2.hconcat(r) for r in tiles]
        full_image = cv2.vconcat(rows)
        out_path = os.path.join(output_folder, f"reconstructed_{frame_id:05d}.png")
        cv2.imwrite(out_path, full_image)
    except Exception as e:
        print(f"❌ Lỗi khi ghép frame {frame_id}: {e}")

print("✅ Đã ghép xong toàn bộ ảnh.")
