from typing import List, Dict, Any
from datasets import load_dataset
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import time

class LLMModel:
    """Main LLM Model class for loading models, tokenizers, and datasets"""
    
    def __init__(self, model_name: str, quantization_config: BitsAndBytesConfig):
        """
        Initialize LLMModel with a specific model name
        
        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-32B")
        """
        print(f"Initializing LLMModel with: {model_name}")
        self.quantization_config = quantization_config
        self.model_name = model_name
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        
        
    
    def load_tokenizer(self) -> AutoTokenizer:
        """Load and return the tokenizer for the model"""
        print(f"Loading tokenizer for {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
        return tokenizer
    
    def load_model(self) -> AutoModelForCausalLM:
        """Load and return the model"""
        print(f"Loading model {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=self.quantization_config
        )
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "model_name": self.model_name,
            "tokenizer_loaded": self.tokenizer is not None,
            "model_loaded": self.model is not None,
        }
        
        if self.tokenizer:
            info["vocab_size"] = self.tokenizer.vocab_size
            info["pad_token"] = self.tokenizer.pad_token
        
        return info


    def inference_sample(self, query):
        """Sample inference from the model"""
        print(f"Sampling inference from the model")
        print(f"Model: {self.model_name}")
        print(f"Model info: {self.get_model_info()}")
        print(f"Query Keys: {query.keys()}")
        time_start = time.time()
        inputs = self.tokenizer([query['Question'] + self.tokenizer.eos_token], return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs.input_ids, max_new_tokens=1000)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
        return response








