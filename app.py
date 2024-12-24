import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import gc
import logging
import re

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('qwen_vl_ocr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPU内存管理函数
def clean_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU内存清理 - 已用内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# 全局加载模型和处理器
try:
    logger.info("开始加载模型...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,  # 使用半精度浮点数
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    raise

def preprocess_image(image, max_resolution=(1024, 1024)):
    """
    如果输入图像分辨率过高，则进行压缩并保持宽高比。
    :param image: PIL.Image 对象
    :param max_resolution: 最大分辨率 (宽, 高)
    :return: 处理后的 PIL.Image 对象
    """
    if image.width > max_resolution[0] or image.height > max_resolution[1]:
        logger.info(f"原始图像分辨率过高 ({image.width}x{image.height})，正在压缩...")
        image.thumbnail(max_resolution, Image.Resampling.LANCZOS)  # 使用 thumbnail 保持宽高比
        logger.info(f"图像已压缩为 {image.width}x{image.height}")
    return image

def generate_prompt():
    return """
请从这张图片中提取试题内容，包括：
1. 试题的题号
2. 试题的完整文本
3. 试题的所有选项（如果有）
请仅输出图片中的文字内容，不要输出提示词本身。
"""

def clean_output_text(text):
    text = re.sub(r'请从这张图片中提取试题内容.*?请仅输出图片中的文字内容，不要输出提示词本身。', '', text, flags=re.DOTALL)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    return text.strip()

def extract_text_from_image(image, custom_prompt):
    try:
        clean_gpu_memory()
        image = preprocess_image(image)  # 调用图片预处理函数
        prompt = custom_prompt if custom_prompt else generate_prompt()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        logger.debug(f"处理后的图片输入: {image_inputs}")

        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        # 检查输入张量是否存在非法值
        if torch.isnan(inputs.input_ids).any() or torch.isinf(inputs.input_ids).any():
            logger.error("输入张量中存在非法值（NaN 或 Inf）")
            raise ValueError("输入张量中存在非法值（NaN 或 Inf）")

        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=2,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=100,
                repetition_penalty=1.2
            )
        except RuntimeError as e:
            logger.error(f"CUDA 错误: {e}")
            if "device-side assert triggered" in str(e):
                logger.error("可能是输入数据或生成参数导致的错误，请检查输入张量和生成配置。")
            raise

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        logger.debug(f"生成的原始文本: {output_text}")
        cleaned_text = clean_output_text(output_text)
        logger.debug(f"清理后的文本: {cleaned_text}")
        
        return cleaned_text if cleaned_text.strip() else "未能提取到有效文字内容，请检查图片质量或尝试调整参数。"

    except Exception as e:
        logger.error(f"文字提取错误: {e}", exc_info=True)
        return f"处理出错: {e}\n请检查图像并重试。"

# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## 中文试卷OCR系统")
    gr.Markdown("### 支持识别中文试卷内容，自动保持格式")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传试卷图片", interactive=True)
            custom_prompt = gr.Textbox(label="自定义提示词（可选）", placeholder="请输入特定的提取要求...", lines=2)
        with gr.Column():
            text_output = gr.Textbox(label="识别结果", lines=25)

    extract_button = gr.Button("开始识别")
    extract_button.click(extract_text_from_image, inputs=[image_input, custom_prompt], outputs=text_output)

    gr.Markdown("""
    ### 使用说明：
    1. 上传需要识别的试卷图片
    2. 可以输入自定义提示词来优化识别效果
    3. 点击"开始识别"按钮进行文字提取
    """)

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=7860, debug=True)
