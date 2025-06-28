import json
import pandas as pd
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class DatasetTemplate:
    """Template configuration for different LLM formats"""
    system_prompt: str
    user_prefix: str
    assistant_prefix: str
    user_suffix: str
    assistant_suffix: str
    conversation_separator: str = "\n\n"


class BaseConverter(ABC):
    """Abstract base class for dataset converters"""
    
    def __init__(self, template: DatasetTemplate):
        self.template = template
    
    @abstractmethod
    def convert_sample(self, sample: Dict) -> str:
        """Convert a single sample to the target format"""
        pass
    
    def convert_dataset(self, dataset: List[Dict]) -> List[str]:
        """Convert entire dataset"""
        return [self.convert_sample(sample) for sample in dataset]


class AlpacaConverter(BaseConverter):
    """Converter for Alpaca format"""
    
    def __init__(self):
        template = DatasetTemplate(
            system_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
            user_prefix="### Instruction:\n",
            assistant_prefix="### Response:\n",
            user_suffix="",
            assistant_suffix=""
        )
        super().__init__(template)
    
    def convert_sample(self, sample: Dict) -> str:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        if input_text:
            instruction = f"{instruction}\n\nInput: {input_text}"
        
        return f"{self.template.user_prefix}{instruction}{self.template.user_suffix}{self.template.conversation_separator}{self.template.assistant_prefix}{output}{self.template.assistant_suffix}"


class ChatMLConverter(BaseConverter):
    """Converter for ChatML format"""
    
    def __init__(self):
        template = DatasetTemplate(
            system_prompt="You are a helpful assistant.",
            user_prefix="<|im_start|>user\n",
            assistant_prefix="<|im_start|>assistant\n",
            user_suffix="<|im_end|>",
            assistant_suffix="<|im_end|>"
        )
        super().__init__(template)
    
    def convert_sample(self, sample: Dict) -> str:
        messages = sample.get('messages', [])
        if not messages:
            return ""
        
        formatted_messages = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                formatted_messages.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == 'user':
                formatted_messages.append(f"{self.template.user_prefix}{content}{self.template.user_suffix}")
            elif role == 'assistant':
                formatted_messages.append(f"{self.template.assistant_prefix}{content}{self.template.assistant_suffix}")
        
        return "\n".join(formatted_messages)


class LlamaConverter(BaseConverter):
    """Converter for Llama format"""
    
    def __init__(self):
        template = DatasetTemplate(
            system_prompt="",
            user_prefix="[INST] ",
            assistant_prefix="[/INST] ",
            user_suffix="",
            assistant_suffix=""
        )
        super().__init__(template)
    
    def convert_sample(self, sample: Dict) -> str:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        if input_text:
            instruction = f"{instruction}\n\n{input_text}"
        
        return f"{self.template.user_prefix}{instruction}{self.template.user_suffix}{output}{self.template.assistant_suffix}"


class Qwen3Converter(BaseConverter):
    """Converter for Qwen3 format with medical reasoning style"""
    
    def __init__(self):
        # Using a generic template structure, but the actual formatting is handled in convert_sample
        template = DatasetTemplate(
            system_prompt="Below is an instruction that describes a task, paired with an input that provides further context.",
            user_prefix="### Instruction:\n",
            assistant_prefix="### Response:\n",
            user_suffix="",
            assistant_suffix=""
        )
        super().__init__(template)
        
        self.train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thought to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>
{}
</think>
{}"""
    
    def convert_sample(self, sample: Dict) -> str:
        # Handle different possible field names for the medical dataset
        question = sample.get('Question', sample.get('question', sample.get('instruction', '')))
        complex_cot = sample.get('Complex_CoT', sample.get('complex_cot', sample.get('reasoning', sample.get('thinking', ''))))
        response = sample.get('Response', sample.get('response', sample.get('output', sample.get('answer', ''))))
        
        # If it's not a medical format, fall back to standard instruction format
        if not question or not response:
            instruction = sample.get('instruction', '')
            input_text = sample.get('input', '')
            output = sample.get('output', '')
            
            if input_text:
                question = f"{instruction}\n\n{input_text}"
            else:
                question = instruction
            
            response = output
            complex_cot = sample.get('reasoning', '')
        
        return self.train_prompt_style.format(question, complex_cot, response)


class DatasetConverter:
    """Main converter class that handles different formats"""
    
    CONVERTERS = {
        'alpaca': AlpacaConverter,
        'chatml': ChatMLConverter,
        'llama': LlamaConverter,
        'qwen3': Qwen3Converter
    }
    
    def __init__(self, format_type: str = 'alpaca'):
        if format_type not in self.CONVERTERS:
            raise ValueError(f"Unsupported format: {format_type}. Supported formats: {list(self.CONVERTERS.keys())}")
        
        self.converter = self.CONVERTERS[format_type]()
    
    def convert(self, dataset: Union[List[Dict], str]) -> List[str]:
        """
        Convert dataset to target format
        
        Args:
            dataset: Dataset in various formats (list of dicts, JSON file path, or DataFrame)
        
        Returns:
            List of formatted strings
        """
        # Load dataset if it's a file path
        if isinstance(dataset, str):
            with open(dataset, 'r', encoding='utf-8') as f:
                if dataset.endswith('.jsonl'):
                    dataset = [json.loads(line) for line in f if line.strip()]
                else:
                    dataset = json.load(f)
        
        # Convert DataFrame to list of dicts if pandas is available
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_dict('records')  # type: ignore
        
        return self.converter.convert_dataset(dataset)
    
    def save_converted_dataset(self, converted_data: List[str], output_path: str):
        """Save converted dataset to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(item + '\n\n')


def convert_dataset_to_format(dataset: Union[List[Dict], str], 
                            format_type: str = 'alpaca',
                            output_path: Optional[str] = None) -> List[str]:
    """
    Convenience function to convert dataset to specified format
    
    Args:
        dataset: Input dataset
        format_type: Target format ('alpaca', 'chatml', 'llama', 'qwen3')
        output_path: Optional path to save converted dataset
    
    Returns:
        List of converted samples
    """
    converter = DatasetConverter(format_type)
    converted_data = converter.convert(dataset)
    
    if output_path:
        converter.save_converted_dataset(converted_data, output_path)
    
    return converted_data


# Example usage and testing
if __name__ == "__main__":
    # Example dataset with medical reasoning format
    medical_dataset = [
        {
            "Question": "Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?",
            "Complex_CoT": "Okay, let's see what's going on here. We've got sudden weakness in the person's left arm and leg - and that screams something neuro-related, maybe a stroke? But wait, there's more. The right lower leg is swollen and tender, which is like waving a big flag for deep vein thrombosis, especially after a long flight...",
            "Response": "The specific cardiac abnormality most likely to be found in this scenario is a patent foramen ovale (PFO)."
        }
    ]
    
    # Standard dataset format
    standard_dataset = [
        {
            "instruction": "Translate the following to French",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous?"
        }
    ]
    
    # Test different converters
    for format_type in ['alpaca', 'chatml', 'llama', 'qwen3']:
        print(f"\n=== {format_type.upper()} Format ===")
        
        # Use medical dataset for qwen3, standard for others
        test_dataset = medical_dataset if format_type == 'qwen3' else standard_dataset
        
        converted = convert_dataset_to_format(test_dataset, format_type)
        for i, item in enumerate(converted):
            print(f"Sample {i+1}:")
            print(item)
            print("-" * 80)
