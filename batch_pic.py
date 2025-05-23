from qwen_picture_bar import Qwen_pic
import glob       # 引入 glob 模組來查找文件
import math       # 引入 math 模組來計算批次數
import os


# PIC_DIR = "/Users/tycg/Desktop/qwen/frames"  # 圖片所在的目錄


# # --- 準備所有圖片路徑 ---
# print(f"Searching for picture in: {PIC_DIR}")
# all_frame_paths = sorted(glob.glob(os.path.join(PIC_DIR, 'pic_*.jpg')))
# total_images = len(all_frame_paths)

# if total_images == 0:
#     print(f"Error: No images found in {PIC_DIR}")
#     exit()

prompt="lease help me to organize this table into CSV format. Just give me the converted data. No other additional responses are needed."
path1=f"/Users/tycg/Desktop/qwen/testdata/table1.jpg"
path2=f"/Users/tycg/Desktop/qwen/testdata/table2.jpg"

Qwen_pic(prompt,path1)
Qwen_pic(prompt,path2)

# --- 批次 ---
# for i in range(total_images):
#     print(f"Processing in {i} batches of size {total_images}.")
#     #Qwen_pic(i,i)
#     Qwen_pic(i,i)

