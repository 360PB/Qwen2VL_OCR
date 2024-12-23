import gradio as gr
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
