from gemma3 import gemma3_pic
from gemma3_text import gemma3_text

import glob       # 引入 glob 模組來查找文件
import os

# PIC_DIR = "/Users/tycg/Desktop/qwen/frames"  # 圖片所在的目錄 (用於批次處理)
# IMAGE_PATTERN = 'frame_*.jpg'              # 批次處理時尋找的圖片模式
# NUM_IMAGES_TO_PROCESS = 0                # 設定"批次處理"要讀取的圖片數量 (設為 None 或 0 表示讀取全部)
# prompt="請用繁體中文，以描述此圖的內容，後續提供LLM讀取，理解整部影片，不要包含任何英文或解釋。"

# # --- １. 準備批次處理 ---
# print(f"--- 步驟 １: 開始準備批次處理 (從目錄: {PIC_DIR}) ---")
# print(f"正在搜尋圖片於: {PIC_DIR} (模式: {IMAGE_PATTERN})")
# search_path = os.path.join(PIC_DIR, IMAGE_PATTERN) 
# all_frame_paths = sorted(glob.glob(search_path))
# total_available_images = len(all_frame_paths)

# # --- 檢查是否有找到圖片 (用於批次處理) ---
# if total_available_images == 0:
#     print(f"資訊: 在 {PIC_DIR} 中找不到符合 '{IMAGE_PATTERN}' 模式的圖片，將跳過批次處理。")
#     images_to_process_paths = [] # 沒有圖片可處理
# else:
#     print(f"在目錄中找到 {total_available_images} 張圖片。")

#     # --- 根據 NUM_IMAGES_TO_PROCESS 決定實際要"批次處理"的圖片列表 ---
#     if NUM_IMAGES_TO_PROCESS is None or NUM_IMAGES_TO_PROCESS <= 0:
#         images_to_process_paths = all_frame_paths
#         print(f"將批次處理全部 {total_available_images} 張找到的圖片。")
#     elif NUM_IMAGES_TO_PROCESS >= total_available_images:
#         images_to_process_paths = all_frame_paths
#         print(f"設定數量 ({NUM_IMAGES_TO_PROCESS}) >= 找到的數量 ({total_available_images})，將批次處理全部 {total_available_images} 張圖片。")
#     else:
#         images_to_process_paths = all_frame_paths[:NUM_IMAGES_TO_PROCESS]
#         print(f"將批次處理前 {NUM_IMAGES_TO_PROCESS} 張找到的圖片。")

# num_selected_images = len(images_to_process_paths)

# # --- 2. 執行批次處理 ---
# if num_selected_images > 0:
#     print(f"\n--- 步驟 2: 開始批次處理 {num_selected_images} 張圖片 ---")
#     for index, image_path in enumerate(images_to_process_paths):
#         print(f"正在批次處理第 {index + 1}/{num_selected_images} 張: {os.path.basename(image_path)}")
#         try:
#             # *** 直接呼叫外部定義的 gemma3_pic 函數 ***
#             gemma3_pic(prompt, image_path,str(index + 1))

#         except NameError:
#              print(f"錯誤：看起來 gemma3_pic 函數尚未被定義或導入。請先定義或導入該函數。")
#              exit() # 如果函數不存在，無法繼續，直接退出
#         except Exception as e:
#             # 如果處理某張圖片時出錯，印出錯誤訊息並繼續處理下一張
#             print(f"處理批次圖片 {image_path} 時發生錯誤: {e}")
#             # 你可以選擇在這裡中斷迴圈 (break) 或紀錄錯誤稍後處理

#     print(f"\n--- 完成批次處理 {num_selected_images} 張圖片 ---")

# elif total_available_images > 0:
#      # 如果找到了圖片，但根據設定選出的數量是 0
#      print("\n--- 步驟 2: 根據 NUM_IMAGES_TO_PROCESS 的設定，沒有選定用於批次處理的圖片。 ---")

# # --- 程式結束 ---
# print("\n--- 所有處理流程完成 ---")




#下面這ＰＡＲＴ因某種原因暫時無法執行（可能設定要調，應該沒超出ＴＯＫＥＮ數量，線上版ＧＥＭＭＡ３可用）
filename = "gemma3_pic_output.txt"
text = "" 
try:
    # 'encoding='utf-8'' 是常用的文字編碼，如果你的檔案是其他編碼 (例如 big5)，請修改這裡
    with open(filename, 'r', encoding='utf-8') as f:
        # f.read() 會讀取檔案的全部內容，並回傳一個字串
        text = f.read()

    # 3. (選擇性) 印出讀取到的內容，確認是否成功
    print(f"成功讀取檔案 '{filename}' 的內容。")
except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{filename}'。請確認檔案名稱和路徑是否正確。")
except Exception as e:
    print(f"讀取檔案時發生錯誤：{e}")
gemma3_text("我把短片切分成多張圖片影格，給AI描述，根據這段AI圖片辨識描述。你描述一下這部短片"+text)