import cv2
import os
import sys
sys.path.append('/content/drive/MyDrive')

from VideoUpScaling.utils import utils_logger
from VideoUpScaling.utils import utils_image as util
from matplotlib import image
from matplotlib import pyplot as plt
from VideoUpScaling.BSRGANModule import BSRGAN
import numpy as np
# === Cấu hình ===
# video_path = "datatest/10.72.216.20_.mp4"          # Đường dẫn video gốc
# output_dir = "output_frames_tiles"     # Thư mục lưu ảnh
# frame_step = 57                   # Mỗi bao nhiêu frame thì lấy (1 = mọi frame)

resize_size = (960, 960)
# os.makedirs(output_dir, exist_ok=True)

# === Mở video ===
# cap = cv2.VideoCapture(video_path)
# frame_id = 0
class TilesProcessor:
    def __init__(self, resize_size,mode, output_dir):
        self.resize_size = resize_size
        self.output_dir = output_dir
        self.mode=mode
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_id = 0
    def TilesDivision(self,input,model):
        Image = cv2.imread(input, cv2.IMREAD_COLOR)  # đọc và chuyển sẵn về BGR (3 kênh)
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)  # nếu cần RGB (ví dụ cho matplotlib)
        Image = cv2.resize(Image,resize_size)
        H, W,_ = Image.shape
        # === Chia tile ===
        rows, cols = 3, 3
        tile_H, tile_W = H // rows, W // cols
        results = []
        scale = 4
        for r in range(rows):
            for c in range(cols):
                x1 = c * tile_W
                y1 = r * tile_H
                x2 = x1 + tile_W
                y2 = y1 + tile_H
                x_offset = x1 * scale
                y_offset = y1 * scale
                tile = Image[y1:y2, x1:x2]
                tile_name = f"frame_{self.frame_id:05d}_tile_{r}_{c}.png"
                tile_path = os.path.join(self.output_dir, tile_name)
                
                if self.mode=="train":
                    cv2.imwrite(tile_path, tile)
                    print(f"Saved tile: {tile_path}")
                    self.frame_id += 1
                elif self.mode == "test":
                 # === Bước Upscale (nếu cần, ví dụ dùng cv2.resize) ===
                    Enhancer=BSRGAN(model_names=["BSRGAN"],scale_factor=4)
                    tile=Enhancer.Enhance(input=tile)
                    print("Tile shape after SR:", tile.shape)
                    # tile = np.ascontiguousarray(tile)
                    # Nếu tile hợp lệ thì mới model()
                    result = model(tile, verbose=False)[0]

                      # Vẽ nếu có box
                    if result is not None and result.boxes is not None and len(result.boxes) > 0:
                        plotted = result.plot()
                        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                        plt.imshow(plotted)
                        plt.axis('off')
                        plt.show()
                    else:
                        print("Không có kết quả hoặc tile không hợp lệ.")
                    # plt.imshow(result)
                    # plt.axis('off')
                    # plt.show()
                 # === Dịch kết quả về tọa độ toàn cục ===
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        conf = float(box.conf.item())
                        print("toa độ bounding box" ,box)
                        # Box: xyxy
                        xA, yA, xB, yB = box.xyxy[0]
                        #adjust globally
                        xA_global = int(xA + x_offset)
                        yA_global = int(yA + y_offset)
                        xB_global = int(xB + x_offset)
                        yB_global = int(yB + y_offset)
                    
                        results.append((xA_global, yA_global, xB_global, yB_global, cls, conf))
                    # === Vẽ kết quả ===
        for (xA, yA, xB, yB, cls, conf) in results:
            Image = Enhancer.Enhance(input=Image)
            cv2.rectangle(Image, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 2)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(Image, label, (int(xA), int(yA) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # === Lưu ảnh sau khi vẽ box ===
        save_path = os.path.join(self.output_dir, f"result_{self.frame_id:05d}.jpg")
        cv2.imwrite(save_path, Image)
        print(f"✅ Đã lưu kết quả tại: {save_path}")
        plt.imshow(Image)
        plt.axis('off')
        plt.show()
        return results

    def MergingTile(input_folder, output_folder):
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


