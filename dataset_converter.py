import json
import pandas as pd
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from datasets import load_dataset



class CustomDataset:
    def __init__(self, dataset_name: str, subset: str = "en", tokenizer: AutoTokenizer = None, dataset_format: str = "qwen3"):
        self.dataset_name = dataset_name
        self.subset = subset
        self.train_dataset = load_dataset(self.dataset_name, self.subset, split="train", trust_remote_code=True)
        self.test_dataset = load_dataset(self.dataset_name, self.subset, split="test", trust_remote_code=True)
        self.prompt_style = self.train_prompt_style(dataset_format)
        self.tokenizer = tokenizer
    

    def train_prompt_style(self, dataset_format: str):
        if dataset_format == "qwen3":
            train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
                Write a response that appropriately completes the request. 
                Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

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
            return train_prompt_style
        else:
            raise ValueError(f"Invalid dataset format: {dataset_format}")
    
    def format_prompt(self, examples):
        inputs = examples["Question"]
        complex_cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []

        for question, cot, response in zip(inputs, complex_cots, outputs):
            # Append the EOS token to the response if it's not already there
            if not response.endswith(self.tokenizer.eos_token):
                response += self.tokenizer.eos_token
            text = self.prompt_style.format(question, cot, response)
            texts.append(text)
        return {"text": texts}

    
    def get_train_dataset(self):
        return self.train_dataset.map(self.format_prompt, batched=True)
    
    def get_test_dataset(self):
        return self.test_dataset.map(self.format_prompt, batched=True)



        