from qwen_vedio_32b import Qwen_vedio
from Tarsier2_Recap_7b import Tarsier2
import glob       # 引入 glob 模組來查找文件
import math       # 引入 math 模組來計算批次數
import os

# --- 設定 ---
BATCH_SIZE = 15         # 設定每個批次處理的圖片數量
FRAME_DIR = "/Users/tycg/Desktop/qwen/frames"  # 圖片所在的目錄


# --- 準備所有圖片路徑 ---
print(f"Searching for frames in: {FRAME_DIR}")
all_frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, 'frame_*.jpg')))
total_images = len(all_frame_paths)

if total_images == 0:
    print(f"Error: No frame images (frame_*.jpg) found in {FRAME_DIR}")
    exit()
# --- 計算批次數 ---
num_batches = math.ceil(total_images / BATCH_SIZE)
print(f"Processing in {num_batches} batches of size {BATCH_SIZE}.")


for i in range(num_batches):
    # Qwen_vedio(i,i)
    Tarsier2(i,i)

