# myenv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import traceback  # 引入 traceback 以便打印詳細錯誤
import os         # 引入 os 模組來處理文件路徑
import glob       # 引入 glob 模組來查找文件

# --- 設定 ---
BATCH_SIZE = 15         # 設定每個批次處理的圖片數量
FRAME_DIR = "/Users/tycg/Desktop/qwen/frames"  # 圖片所在的目錄
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT_TEXT = "請使用繁體中文描述這些圖片的內容，並總結這段影片。" # 使用繁體中文提示詞

def Qwen_vedio(batchnum,pic_number):

    # --- 模型與處理器加載 ---
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "auto",
            attn_implementation="eager",
            device_map="mps",
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        exit()

    print(f"Loading processor: {MODEL_PATH}")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
        print("Processor loaded successfully.")
    except Exception as e:
        print(f"Error loading processor: {e}")
        traceback.print_exc()
        exit()

    # --- 設備檢查 ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS ({model.device})") # 模型已透過 device_map 加載
    else:
        print("Warning: MPS not available, falling back to CPU.")
        device = torch.device("cpu")
        print("Moving model to CPU...")
        try:
            model = model.to(device) # 如果 device_map 沒生效或 MPS 不可用，手動移到 CPU
            print("Model moved to CPU.")
        except Exception as e:
            print(f"Error moving model to CPU: {e}")
            traceback.print_exc()
            exit()

    # --- 準備所有圖片路徑 ---
    print(f"Searching for frames in: {FRAME_DIR}")
    all_frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, 'frame_*.jpg')))
    total_images = len(all_frame_paths)

    if total_images == 0:
        print(f"Error: No frame images (frame_*.jpg) found in {FRAME_DIR}")
        exit()

    print(f"Found {total_images} frames.")
    # --------------------------------------------------------------------------------

    # --- display批次數 ---
    print(f"\n--- Processing Batch {batchnum} ---")

    # --- 取得當前批次的圖片路徑 ---
    start_index = pic_number * BATCH_SIZE
    end_index = min((pic_number + 1) * BATCH_SIZE, total_images)
    current_batch_frames = all_frame_paths[start_index:end_index]
    print(f"Processing frames {start_index + 1} to {end_index}")

    if not current_batch_frames:
        print("Warning: Empty batch, skipping.")
            

    # --- 準備當前批次的輸入 Messages ---
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video", # 即使是多張圖片，這裡類型仍用 video 配合舊版 process_vision_info
                        "video": current_batch_frames,
                    },
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            }
        ]

    # --- 處理輸入 ---
    print("Processing messages for batch...")
    try:
        text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # 使用舊的 process_vision_info 調用方式
        image_inputs, video_inputs = process_vision_info(messages)

            # 調用 processor 時不傳遞 video_kwargs
        inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs, # 這裡的 video_inputs 實際上是處理後的圖片
                padding=True,
                return_tensors="pt",
            )
        print("Input processing successful.")

    except Exception as e:
        print(f"Error processing inputs for batch {batchnum}: {e}")
        traceback.print_exc()
        print("Skipping this batch.")

        # --- 移動輸入到目標設備 ---
    try:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        print("Inputs moved successfully.")
    except Exception as e:
        print(f"Error moving inputs to {device} for batch {batchnum}: {e}")
        traceback.print_exc()
        print("Skipping this batch.")      
    
        # --- 推理生成 ---
    try:
        print("processing start .......")
        with torch.no_grad(): # 在推理時使用 no_grad 可以節省記憶體
            generated_ids = model.generate(**inputs, max_new_tokens=512) # 增加 token 數量以獲得更詳細描述

        print("Generation completed, decoding output...")
        generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
        output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        batch_result_text = output_text[0]
        print(f"--- Batch {batchnum} Output ---")
        print(batch_result_text)
        print(f"--- End Batch {batchnum} Output ---")
    except Exception as e:
        print(f"\n {e}")       

        # 可選：將結果寫入文件
    try:
        output_filename = "qwen_vl_batch_output.txt"
        with open(output_filename, "a+", encoding="utf-8") as f:
            f.write(f"\nFrames {start_index + 1}-{end_index}")
            f.write(batch_result_text)
        print(f"\nConsolidated results also saved to: {output_filename}")
    except Exception as e:
        print(f"\nError writing results to file: {e}")        

        

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

