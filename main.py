#!/usr/bin/env python3
"""
TuneLLM - Main entry point for LLM fine-tuning pipeline
"""

import time
from LLMModel import LLMModel
import torch
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, GRPOTrainer, DPOTrainer
from transformers import TrainingArguments
import gc
from dataset_converter import CustomDataset
from transformers import DataCollatorForLanguageModeling



# Configuration
MODEL_NAME = "Qwen/Qwen3-4B"
DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"
FINETUNING_METHOD = "sft"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)





def train_model(model, tokenizer, dataset):
    print("Training model")
    training_args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        logging_steps=0.2,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        group_by_length=True,
        report_to="none"
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    if FINETUNING_METHOD == "sft":
        print("Training with SFT")
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            data_collator=data_collator
        )
    elif FINETUNING_METHOD == "dpo":
        print("Training with DPO")

    elif FINETUNING_METHOD == "grpo":
        print("Training with GRPO")

    else:
        raise ValueError(f"Invalid finetuning method: {FINETUNING_METHOD}")

    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = False
    trainer.train()



def benchmark_model(model, tokenizer, dataset):
    print("Benchmarking model")
    print("Model: ", model.model_name)
    print("Model info: ", model.get_model_info())
    print("Dataset: ", dataset)
    print("Tokenizer: ", tokenizer)



def main():
    """Main execution function"""
    print("=" * 60)
    print("TuneLLM - LLM Fine-tuning Pipeline")
    print("=" * 60)
    
    try:
        # Initialize model
        print(f"\n1. Initializing model: {MODEL_NAME}")
        model = LLMModel(MODEL_NAME)
        
        # Show model info
        model_info = model.get_model_info()
        print(f"Model info: {model_info}")
        

        # Load dataset
        print(f"\n2. Loading dataset: {DATASET_NAME}")
        dataset = CustomDataset(DATASET_NAME, tokenizer=model.tokenizer, dataset_format="qwen3")        
        train_dataset = dataset.get_train_dataset() 
        test_dataset = dataset.get_test_dataset()
        print(f"Size of train dataset: {len(train_dataset)}")
        print(f"Size of test dataset: {len(test_dataset)}")

        

        # Sample inference with converted prompt
        # print(f"\n4. Sampling inference from the model")
        # outputs = model.inference_sample(train_dataset[0])
        # print(f"Inference completed")
        
        
        # Apply LoRA
        # print(f"\n6. Applying LoRA configuration...")
        # peft_model = get_peft_model(model.model, peft_config)  # type: ignore
        # peft_model.print_trainable_parameters()  # type: ignore
        
        # Train the model
        # print(f"\n7. Starting training...")
        # train_model(peft_model, model.tokenizer, train_dataset)


        # benchmark_model(peft_model, model.tokenizer, train_dataset)
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("Model and tokenizer are loaded in memory.")
    print("=" * 60)


if __name__ == "__main__":
    main()









