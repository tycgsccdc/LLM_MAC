import torch
import transformers
import os # 引入 os 模块方便路径操作

# --- 重要：请将下面的路径修改为你实际存放模型文件的本地文件夹路径 ---
# 例如: "/Users/your_username/Desktop/models/Llama-3_3-Nemotron-Super-49B-v1"
# 或者 "D:\\models\\Llama-3_3-Nemotron-Super-49B-v1" (Windows系统)
local_model_path = "/Users/tycg/Desktop/qwen/NV" # <<<--- 修改这里！

def nemotron(input):
    # 检查路径是否存在，给用户一个提示
    if not os.path.isdir(local_model_path):
        raise FileNotFoundError(
            f"错误：本地模型路径 '{local_model_path}' 不存在或不是一个文件夹。\n"
            f"请确保你已经将 'nvidia/Llama-3_3-Nemotron-Super-49B-v1' 模型的所有文件下载到了这个路径，"
            f"并且路径填写正确。"
        )
    print(f"将从本地路径加载模型和分词器: {local_model_path}")

    # 模型加载参数
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,  # 如果模型包含自定义代码，这个仍然需要
        "device_map": "mps"
    }

    try:
        # 从本地路径加载分词器
        print("正在加载分词器 (tokenizer)...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True # 有些分词器也可能需要这个
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id # 很多模型需要设置这个
        print("分词器加载成功！")

        # 创建 pipeline，并从本地路径加载模型
        # 注意：当 model 参数是本地路径时，pipeline 会尝试从该路径加载模型
        print("正在创建 pipeline 并加载模型...")
        # 为了减少初次运行的内存和时间消耗，可以将 max_new_tokens 暂时调小
        # 确认流程跑通后再调回 32768
        pipeline_instance = transformers.pipeline(
        "text-generation",
        model=local_model_path,    # <--- 使用本地路径
        tokenizer=tokenizer,       # <--- 传入已经加载好的分词器
        max_new_tokens=4096,        # <--- 暂时调小以便快速测试，之后可以改回 32768
        temperature=0.8,
        top_p=0.95,
        **model_kwargs
        )
        print("Pipeline 和模型加载成功！")


        # 假設這是您想要載入的內容
        input_text = input


        thinking = "on"

        prompt_messages = [
            {"role": "system", "content": f"detailed thinking {thinking},use Traditional Chinese answer"},
            {"role": "user", "content": input_text}  # 在這裡使用變數
        ]

        print(f"正在使用 pipeline 生成文本，输入: {prompt_messages}")
        results = pipeline_instance(prompt_messages)
        print("生成结果:")
        print(results)
        return results[0]['generated_text'][-1]['content']

    except Exception as e:
        print(f"在加载或运行模型时发生错误: {e}")
