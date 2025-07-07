# 🎯 TuneLLM

> **Empowering individuals to fine-tune Large Language Models with custom datasets using cutting-edge techniques like LoRA, QLoRA, GRPO, DPO, and more. Compare accuracies, training speeds, and inference performance effortlessly.**

## 📊 Dataset Conversion

This project includes a flexible dataset converter (`dataset_converter.py`) that supports multiple conversation formats commonly used in LLM fine-tuning.

### 🎨 Supported Formats

| Format | Description |
|--------|-------------|
| **🦙 Alpaca** | Instruction-following format with instruction, input, and output fields |
| **💬 ChatML** | Multi-turn conversation format with system, user, and assistant messages |
| **🦙 Llama** | Llama-specific instruction format |
| **🌟 Qwen 3** | Qwen 3 instruction format |

### 🚀 Usage

*[Usage instructions will be added here]*

---

## 📈 Results

### 🏆 Qwen-4B Quantization Benchmark

| Quantization | Precision | Training Time (Epoch 1) | Final Loss | Final Accuracy | GPU Memory |
|:------------:|:---------:|:----------------------:|:----------:|:-------------:|:----------:|
| **FP16** | Half | 4h 59m | 1.1642 | **86.7%** | 8.2 GB |
| **BF16** | Brain | 8h 20m | 1.88 | **87.0%** | 8.5 GB |
| **INT8** | 8-bit | 6h 10m | 2.05 | **84.1%** | 6.1 GB |
| **INT8** [batch_size=2] | 8-bit | 1h 33m | 2.05 | **84.1%** | 39 GB |
| **INT4** | 4-bit | 4h 45m | 2.35 | **81.3%** | 3.9 GB |

> 💡 **Key Insights:**
> - **Best Accuracy**: BF16 (87.0%)
> - **Fastest Training**: INT8 with batch_size=2 (1h 33m)
> - **Lowest Memory**: INT4 (3.9 GB)
> - **Best Balance**: FP16 (86.7% accuracy, reasonable time & memory)
