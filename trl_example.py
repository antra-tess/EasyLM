#!/usr/bin/env python
# coding=utf-8

"""
Example script for using TRL (Transformer Reinforcement Learning) with Gemma-3.
This is a starting point for RLHF (Reinforcement Learning from Human Feedback).
"""

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import (
    SFTTrainer,
    RewardTrainer,
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)

# Configuration
MODEL_NAME = "google/gemma-3-27b-pt"
REWARD_MODEL_NAME = "google/gemma-3-8b-pt"  # Smaller model for reward modeling
OUTPUT_DIR = "/mnt/disk2/gemma_rlhf_output"
DATASET_PATH = "/mnt/disk2/conversations_all.jsonl"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tokenizer_and_model(model_name, load_in_4bit=True, for_reward_model=False):
    """Load the tokenizer and model with quantization"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization
    compute_dtype = torch.float16
    quant_config = None
    
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    # Set appropriate model class based on usage
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    
    # Prepare model for training if using quantization
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    if "gemma" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Add value head for PPO if needed
    if for_reward_model:
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
    else:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
    
    model = get_peft_model(model, peft_config)
    
    return tokenizer, model

def run_sft():
    """Run Supervised Fine-Tuning with TRL"""
    print("Starting SFT training...")
    
    tokenizer, model = load_tokenizer_and_model(MODEL_NAME)
    
    # Use TRL's SFTTrainer
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/sft",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        save_steps=500,
        logging_steps=50,
        bf16=True,
        report_to="wandb",
        run_name="gemma-3-27b-sft-trl",
    )
    
    # Load dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=1024,
        dataset_text_field="output",  # Adjust based on your dataset
        packing=False,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(f"{OUTPUT_DIR}/sft/final")
    print("SFT training complete!")

def prepare_comparison_dataset():
    """Prepare a comparison dataset for reward modeling"""
    # This is a simplified example - in practice you would need a dataset
    # with pairs of responses where one is preferred over the other
    
    # Example format for reward modeling:
    # {
    #   "input": "What is the capital of France?",
    #   "chosen": "The capital of France is Paris, which is known for the Eiffel Tower and Louvre Museum.",
    #   "rejected": "I think it's Paris but I'm not sure."
    # }
    
    # Load your comparison dataset here
    # ...
    
    return "comparison_dataset_path"

def train_reward_model():
    """Train a reward model using TRL"""
    print("Starting reward model training...")
    
    # Load a smaller model for the reward model
    tokenizer, model = load_tokenizer_and_model(REWARD_MODEL_NAME, for_reward_model=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/reward_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        save_steps=500,
        logging_steps=50,
        bf16=True,
        report_to="wandb",
        run_name="gemma-3-8b-reward-model",
    )
    
    # Load comparison dataset
    comparison_dataset_path = prepare_comparison_dataset()
    dataset = load_dataset(comparison_dataset_path, split="train")
    
    # Initialize reward trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(f"{OUTPUT_DIR}/reward_model/final")
    print("Reward model training complete!")

def run_rlhf():
    """Run RLHF using PPO with TRL"""
    print("Starting RLHF (PPO) training...")
    
    # Load SFT model
    tokenizer, sft_model = load_tokenizer_and_model(f"{OUTPUT_DIR}/sft/final")
    
    # Convert to PPO model with value head
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model)
    
    # Create reference model
    ref_model = create_reference_model(ppo_model)
    
    # Load reward model
    reward_model_path = f"{OUTPUT_DIR}/reward_model/final"
    _, reward_model = load_tokenizer_and_model(reward_model_path, for_reward_model=True)
    
    # PPO configuration
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=8,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.1,
        kl_penalty="kl",
        seed=42,
        use_score_scaling=True,
        use_score_norm=True,
        score_clip=None,
    )
    
    # Load prompt dataset
    prompt_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=prompt_dataset,
        data_collator=None,
    )
    
    # Training loop
    for epoch in range(3):
        print(f"Starting epoch {epoch+1}/3")
        
        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            if batch_idx >= 100:  # Limit number of batches per epoch
                break
                
            # Get responses from the model
            query_tensors = [ppo_trainer.tokenizer.encode(prompt, return_tensors="pt") for prompt in batch["input"]]
            response_tensors = []
            
            for query in query_tensors:
                response = ppo_trainer.generate(query, max_new_tokens=128)
                response_tensors.append(response)
            
            # Compute rewards
            texts = [
                ppo_trainer.tokenizer.decode(r.squeeze())
                for r in response_tensors
            ]
            
            rewards = []
            with torch.no_grad():
                for text in texts:
                    inputs = ppo_trainer.tokenizer(text, return_tensors="pt").to(reward_model.device)
                    reward = reward_model(**inputs).logits[0].detach()
                    rewards.append(reward)
            
            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Stats: {stats}")
        
        # Save checkpoint
        ppo_trainer.save_pretrained(f"{OUTPUT_DIR}/rlhf/epoch_{epoch+1}")
    
    print("RLHF training complete!")

if __name__ == "__main__":
    # Uncomment the function you want to run
    # run_sft()
    # train_reward_model()
    # run_rlhf()
    print("Please uncomment the function you want to run in the script.") 