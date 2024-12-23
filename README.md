# Qwen2VL_OCR

## 简介

Qwen2VL_OCR 是一个基于 **Qwen2-VL-2B-Instruct** 模型的 OCR 应用，支持从图像中提取文字并尽量保留格式。用户可以通过自定义提示词来调整模型的行为。

---

## 功能

- **上传图片**：支持上传包含文字的图片（如扫描文档、截图等）。
- **自定义提示词**：用户可以输入自定义提示词，调整模型的提取行为。
- **文字提取**：模型会提取图片中的文字并尽量保留原始格式。

---

## 安装

1. 克隆项目：
   ```bash
   git clone https://github.com/your-username/Qwen2VL_OCR.git
   cd Qwen2VL_OCR

1. 安装依赖：

   ```
   pip install -r requirements.txt
   ```

2. 下载 **Qwen2-VL-2B-Instruct** 模型权重。首次运行时，模型会自动下载。

------

## 使用

运行应用程序：

```
python app.py
```

Gradio 会启动一个本地 Web 服务器。打开终端中提供的 URL（如 `http://127.0.0.1:7860`）即可访问界面。

------

## 示例

### 示例输入

上传图片（如 `examples/example1.jpg`）并输入提示词：

```
提取图像中的文字并保持格式。
```

### 示例输出

```
这是从图像中提取的文字示例。
模型会尽量保留原始格式。
```

以下是优化后的 **Qwen2VL_OCR** 项目代码和说明，支持页面自定义配置提示词，并在 `README.md` 中提供中英文双语支持（默认中文）。

------

### **项目结构**

```
TEXTQwen2VL_OCR/
├── app.py                # 主程序文件
├── requirements.txt      # 项目依赖
├── README.md             # 项目说明文档（中英文）
└── examples/             # 示例图片文件夹
    ├── example1.jpg
    ├── example2.jpg
```

------

### **1. 主程序文件：`app.py`**

```
PYTHONimport gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# 加载模型和处理器
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 定义处理函数
def extract_text_from_image(image, custom_prompt):
    """
    从图像中提取文字并保持格式，支持自定义提示词。
    """
    try:
        # 如果用户未输入自定义提示词，使用默认提示词
        if not custom_prompt.strip():
            custom_prompt = "Extract the text from this image and preserve its format."

        # 准备输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": custom_prompt},
                ],
            }
        ]

        # 处理输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 推理
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
    except Exception as e:
        return f"Error: {e}"

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## Qwen2VL_OCR: 从图像中提取文字")
    gr.Markdown("### 上传图片并输入提示词，模型将提取图片中的文字并尽量保留格式。")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传图像")
            custom_prompt = gr.Textbox(
                label="自定义提示词（可选）",
                placeholder="例如：提取图像中的文字并保持格式。",
                lines=2,
            )
        with gr.Column():
            text_output = gr.Textbox(label="提取的文字", lines=15)

    extract_button = gr.Button("提取文字")
    extract_button.click(
        extract_text_from_image, inputs=[image_input, custom_prompt], outputs=text_output
    )

# 启动 Gradio 应用
if __name__ == "__main__":
    demo.launch()
```

------

### **2. 依赖文件：`requirements.txt`**

```
TXTgradio==3.41.2
transformers==4.34.0
qwen-vl-utils==0.1.0
torch==2.0.1
Pillow==9.5.0
```

------

### **3. 示例图片**

在 `examples/` 文件夹中，放置一些示例图片（如 `example1.jpg` 和 `example2.jpg`），供用户测试。

------

### **4. 项目说明文档：`README.md`**

以下是支持中英文的 `README.md` 文件内容：

```
MARKDOWN# Qwen2VL_OCR

## 简介

Qwen2VL_OCR 是一个基于 **Qwen2-VL-2B-Instruct** 模型的 OCR 应用，支持从图像中提取文字并尽量保留格式。用户可以通过自定义提示词来调整模型的行为。

---

## 功能

- **上传图片**：支持上传包含文字的图片（如扫描文档、截图等）。
- **自定义提示词**：用户可以输入自定义提示词，调整模型的提取行为。
- **文字提取**：模型会提取图片中的文字并尽量保留原始格式。

---

## 安装

1. 克隆项目：
   ```bash
   git clone https://github.com/your-username/Qwen2VL_OCR.git
   cd Qwen2VL_OCR
```

1. 安装依赖：

   ```
   BASH
   
   
   
   pip install -r requirements.txt
   ```

2. 下载 **Qwen2-VL-2B-Instruct** 模型权重。首次运行时，模型会自动下载。

------

## 使用

运行应用程序：

```
BASH



python app.py
```

Gradio 会启动一个本地 Web 服务器。打开终端中提供的 URL（如 `http://127.0.0.1:7860`）即可访问界面。

------

## 示例

### 示例输入

上传图片（如 `examples/example1.jpg`）并输入提示词：

```
TEXT



提取图像中的文字并保持格式。
```

### 示例输出

```
TEXT这是从图像中提取的文字示例。
模型会尽量保留原始格式。
```

------

## 文件结构

```
TEXTQwen2VL_OCR/
├── app.py                # 主程序文件
├── requirements.txt      # 项目依赖
├── README.md             # 项目说明文档（中英文）
└── examples/             # 示例图片文件夹
    ├── example1.jpg
    ├── example2.jpg
```

------

## 注意事项

1. **设备要求**：模型推理需要 GPU 支持。请确保您的环境中有 CUDA 兼容的 GPU 和相关驱动。
2. **提示词优化**：自定义提示词可以显著影响模型的输出效果。根据具体需求调整提示词内容。

------

## 许可证

本项目基于 Apache 2.0 许可证开源。详情请参阅 [LICENSE](LICENSE)。

------

## 致谢

- [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Gradio](https://gradio.app/)