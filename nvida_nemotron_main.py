# 假設這是你的主腳本檔案 (例如 main_processing.py 或 run_files.py)
import os
import nvida_nemotron # 導入修改後的 gemma3 模組

# 導入 transformers 和 torch 用來載入模型
from transformers import AutoProcessor
import torch

def main():
    num_files = 15
    input_file_template = "in/{}.txt"
    output_file_template = "output/{}_nemotron_output.txt"

    # 確保輸出目錄存在 (如果沒有的話)
    if not os.path.exists("output"):
        os.makedirs("output")
        print("建立輸出目錄: output/")

    # --- 在迴圈開始前，只載入模型和 Processor 一次 ---

    print(f"\n開始處理 {num_files} 個檔案...")

    for i in range(4, num_files + 1):
        input_filename = input_file_template.format(i)
        output_filename = output_file_template.format(i)

        print(f"\n正在處理檔案: {input_filename}")

        # 1. 讀取輸入檔案
        try:
            # 使用 'utf-8' 編碼讀取，這是處理中文時的良好實踐
            with open(input_filename, 'r', encoding='utf-8') as infile:
                input_content = infile.read()
            print(f"成功讀取檔案: {input_filename}")
        except FileNotFoundError:
            print(f"錯誤: 找不到輸入檔案 {input_filename}。跳過此檔案。")
            continue
        except Exception as e:
            print(f"讀取檔案 {input_filename} 時發生錯誤: {e}。跳過此檔案。")
            continue

        # 2. 呼叫 Gemma 生成函數，傳入已載入的模型和 Processor
        try:
            # 呼叫修改後的函數，傳入 model, processor 和 input_content
            output = nvida_nemotron.nemotron(input_content)
            print(f"成功為檔案 {input_filename} 生成結果.")
        except Exception as e:
            print(f"呼叫 Gemma 函數處理 {input_filename} 時發生錯誤: {e}。跳過此檔案。")
            continue

        # 3. 寫入輸出檔案
        try:
            # 使用 'utf-8' 編碼寫入
            with open(output_filename, 'w', encoding='utf-8') as outfile:
                outfile.write(output)
            print(f"成功將結果寫入檔案: {output_filename}")
        except Exception as e:
            print(f"寫入檔案 {output_filename} 時發生錯誤: {e}。")

    print("\n所有檔案處理完成。")

if __name__ == "__main__":
    main()