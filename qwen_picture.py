# 官方文件微調後讀取圖片  下線約２０萬ＰＩＸ上限約１００萬ＰＸ
# myenv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 加载模型时设置 attn_implementation="eager" 和 device_map="mps"
model_path = "Qwen/Qwen2.5-VL-32B-Instruct"  # 或者 "Qwen/Qwen2.5-VL-3B-Instruct" 如果你想用 3B 模型测试
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",  # 可以保持 auto，或者尝试 torch.bfloat16
    attn_implementation="eager", # 设置为 eager 注意力实现
    device_map="auto", # 先保持 auto，代码中会进一步指定 mps 或 cpu
    low_cpu_mem_usage=True # 可以添加 low_cpu_mem_usage=True 减少 CPU 内存占用
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True) # 保持 use_fast=True

messages = [
    {
        "role": "user",
        "content": [
             {
                "type": "image",
                "image": "/Users/tycg/Desktop/qwen/testdata/Dog.jpeg", # 确保路径正确
            },
            {"type": "text", "text": "use tradisional chinese Describe the picture"}, # 简化文本
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

# --- Debug print 1: Check image_inputs and video_inputs ---
print("--- Debug print 1: image_inputs and video_inputs ---")
print("image_inputs:", image_inputs)
print("video_inputs:", video_inputs)
print("--- End Debug print 1 ---")

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# --- Debug print 2: Check processor outputs (inputs) ---
print("--- Debug print 2: Processor outputs (inputs) ---")
print("Processor outputs:", inputs)
print("--- End Debug print 2 ---")


if torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to("mps") # 显式将模型移动到 MPS 设备
    inputs = inputs.to("mps") # 显式将输入数据移动到 MPS 设备
    print("Using MPS device.")
elif torch.cuda.is_available(): # 仍然保留 CUDA 检查，以防你在其他 CUDA 环境运行代码
    device = torch.device("cuda")
    model = model.to("cuda") # 显式将模型移动到 CUDA 设备
    inputs = inputs.to("cuda") # 显式将输入数据移动到 CUDA 设备
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    model = model.to("cpu") # 显式将模型移动到 CPU 设备
    inputs = inputs.to("cpu") # 显式将输入数据移动到 CPU 设备
    print("Using CPU device.")


# --- Debug print 3: Check inputs just before model.generate() ---
print("--- Debug print 3: Inputs before model.generate() ---")
print("Inputs keys:", inputs.keys())
for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        print(f"{key} shape: {value.shape}, dtype: {value.dtype}")
print("--- End Debug print 3 ---")


# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)