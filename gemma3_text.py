#這個暫時無法執行（文字理解版地端ＬＬＭ）

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

def gemma3_text(prompt):
    
    model_id = "google/gemma-3-27b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="mps"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": ""}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_k=64)    #這行會報錯
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
                # 可選：將結果寫入文件
    try:
        output_filename = "gemma3_text_output.txt"
        with open(output_filename, "a+", encoding="utf-8") as f:
            f.write(decoded)
        print(f"\nConsolidated results also saved to: {output_filename}")
    except Exception as e:
        print(f"\nError writing results to file: {e}")        
    finally:
        del generation # 確保 generation 被刪除，即使出錯  （記憶體清理）
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




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