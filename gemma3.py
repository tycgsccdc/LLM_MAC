from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch

def gemma3_pic(prompt,picpath,frame):
    model_id = "google/gemma-3-27b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="mps"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful picture label assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": picpath},
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
        generation = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_k=64)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
                # 可選：將結果寫入文件
    try:
        output_filename = "gemma3_pic_output.txt"
        with open(output_filename, "a+", encoding="utf-8") as f:
            f.write("\n第"+frame+"張影格\n"+decoded+"\n")
        print(f"\nConsolidated results also saved to: {output_filename}")
    except Exception as e:
        print(f"\nError writing results to file: {e}")        
    finally:
        del generation # 確保 generation 被刪除，即使出錯  （記憶體清理）
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
