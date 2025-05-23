# myenv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time # 匯入 time 模組

def Qwen_pic(prompt,picpath):
    # 載入模型時設定 attn_implementation="eager" 和 device_map="auto" (後續會根據偵測結果移至 mps, cuda 或 cpu)
    model_path = "Qwen/Qwen2.5-VL-32B-Instruct"  # 或者使用 "Qwen/Qwen2.5-VL-3B-Instruct" 測試 3B 模型
    print(f"正在從 {model_path} 載入模型，請稍候...")
    model_load_start = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",  # 可以保持 auto，或者嘗試 torch.bfloat16
        attn_implementation="eager", # 設定為 eager 注意力實作
        device_map="mps", # 先保持 auto，程式碼中會進一步指定 mps, cuda 或 cpu
        low_cpu_mem_usage=True # 添加 low_cpu_mem_usage=True 以嘗試減少 CPU 記憶體佔用
    )
    model_load_end = time.time()
    print(f"模型載入完成，耗時: {model_load_end - model_load_start:.2f} 秒")

    print(f"正在從 {model_path} 載入處理器...")
    processor_load_start = time.time()
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True) # 保持 use_fast=True 以使用快速處理器
    processor_load_end = time.time()
    print(f"處理器載入完成，耗時: {processor_load_end - processor_load_start:.2f} 秒")


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": picpath, # 確保圖片路徑正確
                },
                {"type": "text", "text": prompt}, # 使用者prompt
            ],
        }
    ]


    # 推理準備
    print("正在準備輸入資料...")
    prep_start = time.time()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # --- 除錯列印 1: 檢查 image_inputs 和 video_inputs ---

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,        # 對輸入進行填充
        return_tensors="pt", # 返回 PyTorch 張量
    )
    prep_end = time.time()
    print(f"輸入資料準備完成，耗時: {prep_end - prep_start:.2f} 秒")

    # --- 除錯列印 2: 檢查處理器輸出 (inputs) ---

    print("正在偵測可用裝置並將模型和資料移至裝置...")
    if torch.backends.mps.is_available():
        print("偵測到 MPS 裝置，正在移動模型和資料...")
        model = model.to("mps") # 明確將模型移動到 MPS 裝置
        inputs = inputs.to("mps") # 明確將輸入資料移動到 MPS 裝置
        print("已使用 MPS 裝置。")


    # --- 除錯列印 3: 檢查 model.generate() 之前的輸入 ---


    # --- 添加計算提示與計時 ---
    print("\n模型正在進行推理計算，請耐心等待...") # <--- 計算開始前的提示訊息
    start_time = time.time() # <--- 記錄開始時間

    # 推理：產生輸出
    # 使用 **inputs 將字典解包為關鍵字參數傳遞給 model.generate
    # max_new_tokens 控制生成文字的最大長度s
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

    end_time = time.time() # <--- 記錄結束時間
    calculation_time = end_time - start_time # <--- 計算耗時
    print(f"推理計算完成！") # <--- 計算完成的提示訊息
    # --- 結束添加的部分 ---


    print("正在解碼輸出...")
    decode_start = time.time()
    # 從產生的 ID 中移除輸入部分的 ID，只保留新生成的 token ID
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # 將修剪後的 ID 批次解碼為文字
    # skip_special_tokens=True：跳過特殊標記（如 [CLS], [SEP] 等）
    # clean_up_tokenization_spaces=False：保留分詞引入的空格（通常推薦用於生成）
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    decode_end = time.time()
    print(f"解碼完成，耗時: {decode_end - decode_start:.2f} 秒")

    print("\n模型輸出:")
    print(output_text[0]) # 通常只有一個輸出，直接列印第一個元素


            # 可選：將結果寫入文件
    try:
        output_filename = "qwen_pic_output.txt"
        with open(output_filename, "a+", encoding="utf-8") as f:
            f.write(output_text[0]+"\n")
        print(f"\nConsolidated results also saved to: {output_filename}")
    except Exception as e:
        print(f"\nError writing results to file: {e}")        


    # --- 列印推理耗時 ---
    print(f"\n模型推理（generate 呼叫）耗時: {calculation_time:.2f} 秒") # <--- 列印計算時間


    # --- 清理記憶體  ---
    del inputs, generated_ids, generated_ids_trimmed, output_text
    if torch.backends.mps.is_available():
            try:
                import gc
                gc.collect()
            except Exception as e_gc:
                print(f"Warning: Error during potential memory cleanup: {e_gc}")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()