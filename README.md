# TuneLLM
This repo basically helps any individual to fientune their LLM with their custom dataset using various methods like LoRA, QLoRA, GRPO, DPO, etc and compare the accuracies and training and inference speeds

## Dataset Conversion

This project includes a flexible dataset converter (`dataset_converter.py`) that supports multiple conversation formats commonly used in LLM fine-tuning.

### Supported Formats

- **Alpaca**: Instruction-following format with instruction, input, and output fields
- **ChatML**: Multi-turn conversation format with system, user, and assistant messages
- **Llama**: Llama-specific instruction format
- **Qwen 3**: Qwen 3 instruction format
### Usage


# Results

## Qwen-4B Quantization Benchmark

| Quantization | Precision | Training Time (Epoch 1) | Final Loss | Final Accuracy | GPU Memory |
|--------------|-----------|--------------------------|------------|----------------|------------|
| FP16         | Half      | 4h 59m                   | 1.1642       | 86.7%          | 8.2 GB     |
| BF16         | Brain     | 8h 20m                   | 1.88       | 87.0%          | 8.5 GB     |
| INT8         | 8-bit     | 6h 10m                   | 2.05       | 84.1%          | 6.1 GB     |
| INT8 [batch_size=2]         | 8-bit     | 1h 50m                   | 2.05       | 84.1%          | 39 GB     |
| INT4         | 4-bit     | 4h 45m                   | 2.35       | 81.3%          | 3.9 GB     |
