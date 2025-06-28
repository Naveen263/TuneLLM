# TuneLLM
This repo basically helps any individual to fientune their LLM with their custom dataset using various methods like LORA, QLORA, GRPO, DPO, etc and compare the accuracies and training and inference speeds

## Dataset Conversion

This project includes a flexible dataset converter (`dataset_converter.py`) that supports multiple conversation formats commonly used in LLM fine-tuning.

### Supported Formats

- **Alpaca**: Instruction-following format with instruction, input, and output fields
- **ChatML**: Multi-turn conversation format with system, user, and assistant messages
- **Llama**: Llama-specific instruction format

### Usage
