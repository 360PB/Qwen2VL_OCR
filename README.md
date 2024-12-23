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