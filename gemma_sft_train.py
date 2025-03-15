#!/usr/bin/env python
# coding=utf-8

import os
import yaml
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    AutoConfig,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from datasets import load_dataset

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        default="google/gemma-3-27b-pt",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name or path if different from model"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8bit mode"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 4bit mode"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading pretrained models"}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention for faster training"}
    )

@dataclass
class LoRAArguments:
    """
    Arguments pertaining to LoRA fine-tuning configuration.
    """
    lora_rank: int = field(
        default=32,
        metadata={"help": "Rank of LoRA matrices"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "Scaling factor for LoRA"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for LoRA"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "List of module names to apply LoRA to"}
    )
    target_modules_for_llama: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "List of module names to apply LoRA to for LLaMA models"}
    )
    target_modules_for_gemma: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        metadata={"help": "List of module names to apply LoRA to for Gemma models"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to the data loading and processing.
    """
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the training data jsonl file"}
    )
    template_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the conversation template yaml file"}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length to use for data processing"}
    )

class JsonlDataset(Dataset):
    """
    Dataset for loading data from jsonl files with a specific conversation template
    """
    def __init__(self, dataset_path, tokenizer, template_yaml=None, max_seq_length=1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Check if dataset file exists
        if not os.path.exists(dataset_path):
            error_msg = f"ERROR: Dataset file not found: {dataset_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Log the absolute path of the dataset file
        abs_dataset_path = os.path.abspath(dataset_path)
        logger.info(f"Loading dataset from absolute path: {abs_dataset_path}")
        
        # Load the template if provided
        self.template = None
        if template_yaml:
            # Check if template file exists
            if not os.path.exists(template_yaml):
                error_msg = f"ERROR: Template file not found: {template_yaml}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Log the absolute path of the template file
            abs_template_path = os.path.abspath(template_yaml)
            logger.info(f"Loading template from absolute path: {abs_template_path}")
            
            with open(template_yaml, 'r') as f:
                self.template = yaml.safe_load(f)
                logger.info(f"Template content (summary): {self.template.keys() if isinstance(self.template, dict) else 'Not a dictionary'}")
                
                # Log the template sequence to help with debugging
                if isinstance(self.template, dict) and 'sequence' in self.template:
                    logger.info(f"Template sequence contains {len(self.template['sequence'])} segments")
                    for i, segment in enumerate(self.template['sequence']):
                        keys = list(segment.keys())
                        logger.info(f"  Segment {i+1} keys: {keys}")
                        # Show an example of the expected format
                        for k in keys:
                            value = segment[k]
                            logger.info(f"  Segment {i+1} {k} format string contains these placeholders: {[p for p in self._find_format_placeholders(value)]}")
        
        # Load and process dataset
        self.examples = []
        try:
            with open(dataset_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            example = json.loads(line)
                            self.examples.append(example)
                        except json.JSONDecodeError:
                            logger.warning(f"Line {line_num}: Could not parse as JSON: {line[:100]}...")
                    
            # Check if dataset is empty
            if len(self.examples) == 0:
                error_msg = f"ERROR: Dataset is empty or contains no valid examples: {dataset_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log sample entries to help with debugging
            logger.info(f"Loaded {len(self.examples)} examples from the dataset")
            if len(self.examples) > 0:
                sample_example = self.examples[0]
                logger.info(f"First example keys: {list(sample_example.keys())}")
                
                # Validate template fields against dataset content
                if self.template and 'sequence' in self.template:
                    logger.info("Validating template fields against dataset content...")
                    required_fields = set()
                    for segment in self.template['sequence']:
                        for content in segment.values():
                            # Extract field names from the format string
                            placeholders = self._find_format_placeholders(content)
                            required_fields.update(placeholders)
                    
                    logger.info(f"Template requires these fields: {required_fields}")
                    
                    # Check first example for required fields
                    missing_fields = [field for field in required_fields if field not in sample_example]
                    if missing_fields:
                        error_msg = f"ERROR: First example is missing these required fields: {missing_fields}"
                        logger.error(error_msg)
                        logger.error(f"Dataset example contains only these fields: {list(sample_example.keys())}")
                        raise KeyError(error_msg)
                    else:
                        logger.info("First example contains all required template fields.")
                        
        except Exception as e:
            error_msg = f"ERROR loading dataset: {str(e)}"
            logger.error(error_msg)
            raise
    
    def _find_format_placeholders(self, format_string):
        """Extract field names from a format string"""
        import re
        # This regex finds all format placeholders like {name} in a string
        return re.findall(r'\{([^{}]*)\}', format_string)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        if self.template:
            # Apply template formatting similar to EasyLM's TextProcessor
            text_parts = []
            loss_mask_parts = []
            
            try:
                for segment in self.template['sequence']:
                    for loss_key, content in segment.items():
                        mask_value = 0.0 if loss_key == 'no_loss' else 1.0
                        try:
                            formatted_text = content.format(**example).replace('\\n', '\n')
                        except KeyError as e:
                            # More detailed error message with example index and missing key
                            missing_key = str(e).strip("'")
                            error_msg = f"Example {idx} is missing required field '{missing_key}' for template formatting"
                            logger.error(error_msg)
                            logger.error(f"Example keys: {list(example.keys())}")
                            logger.error(f"Template content: {content}")
                            raise KeyError(error_msg) from e
                        
                        # Tokenize this part
                        tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
                        text_parts.extend(tokens)
                        loss_mask_parts.extend([mask_value] * len(tokens))
            except Exception as e:
                logger.error(f"Error processing example {idx}: {e}")
                raise
            
            # Rest of the function remains the same
            # Add BOS token if not already present
            if len(text_parts) == 0 or text_parts[0] != self.tokenizer.bos_token_id:
                text_parts.insert(0, self.tokenizer.bos_token_id)
                loss_mask_parts.insert(0, 0.0)  # No loss on BOS token
            
            # Add EOS token
            text_parts.append(self.tokenizer.eos_token_id)
            loss_mask_parts.append(1.0)  # Apply loss on EOS token
            
            # Truncate to max length
            if len(text_parts) > self.max_seq_length:
                text_parts = text_parts[:self.max_seq_length]
                loss_mask_parts = loss_mask_parts[:self.max_seq_length]
            
            # Convert to tensor format required by transformer trainer
            input_ids = torch.tensor(text_parts)
            attention_mask = torch.ones_like(input_ids)
            labels = torch.clone(input_ids)
            
            # Apply loss mask - set tokens with mask=0.0 to -100 in labels
            for i, mask_val in enumerate(loss_mask_parts):
                if mask_val == 0.0:
                    labels[i] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # Default processing if no template is provided
            # This assumes a simple format with instruction, input and output fields
            instruction = example.get("instruction", "")
            model_input = example.get("input", "")
            output = example.get("output", "")
            
            prompt = f"{instruction}\n{model_input}".strip()
            text = f"{prompt}\n{output}"
            
            encodings = self.tokenizer(text, max_length=self.max_seq_length, truncation=True, padding="max_length")
            
            # Create labels: -100 for prompt tokens (no loss), actual token IDs for output tokens
            prompt_len = len(self.tokenizer(prompt, truncation=True).input_ids)
            labels = encodings["input_ids"].copy()
            labels[:prompt_len] = [-100] * prompt_len  # No loss for prompt tokens
            
            return {
                "input_ids": torch.tensor(encodings["input_ids"]),
                "attention_mask": torch.tensor(encodings["attention_mask"]),
                "labels": torch.tensor(labels)
            }

def main():
    parser = HfArgumentParser((ModelArguments, LoRAArguments, DataArguments, TrainingArguments))
    model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info(f"Training arguments: {training_args}")
    
    # Check for required dataset path
    if data_args.dataset_path is None:
        error_msg = "ERROR: No dataset path provided. Please specify --dataset_path"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Check for DeepSpeed compatibility
    deepspeed_fallback = False
    if training_args.deepspeed:
        try:
            # Try to import DeepSpeed to see if it's compatible
            import deepspeed
            logger.info(f"Successfully imported DeepSpeed version: {deepspeed.__version__}")
        except ImportError as e:
            logger.warning(f"Failed to import DeepSpeed: {e}")
            logger.warning("Falling back to regular PyTorch training without DeepSpeed")
            training_args.deepspeed = None
            deepspeed_fallback = True
        except Exception as e:
            logger.warning(f"DeepSpeed initialization error: {e}")
            logger.warning("Falling back to regular PyTorch training without DeepSpeed")
            training_args.deepspeed = None
            deepspeed_fallback = True
    
    # Load pretrained model and tokenizer
    tokenizer_name = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    
    # Try to load tokenizer with different configurations to handle compatibility issues
    try:
        logger.info(f"Loading tokenizer from {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=True,
        )
    except Exception as e:
        logger.warning(f"Failed to load fast tokenizer: {e}")
        logger.info("Trying to load with slow tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=model_args.trust_remote_code,
                use_fast=False,
            )
        except Exception as e2:
            logger.warning(f"Failed to load slow tokenizer: {e2}")
            if "gemma" in tokenizer_name.lower():
                logger.info("Trying with specific Gemma-3 configuration...")
                try:
                    # For Gemma-3, we might need to specify the revision and other parameters
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_name,
                        trust_remote_code=True,
                        use_fast=False,
                        padding_side="right",
                        revision="main",
                    )
                except Exception as e3:
                    raise ValueError(f"Could not load tokenizer after multiple attempts. Last error: {e3}")
            else:
                raise ValueError(f"Could not load tokenizer after multiple attempts. Last error: {e2}")
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization for the model if requested
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    quant_config = None
    
    if model_args.load_in_4bit:
        logger.info("Loading model in 4-bit precision")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif model_args.load_in_8bit:
        logger.info("Loading model in 8-bit precision")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        logger.info(f"Loading model in {compute_dtype} precision (unquantized)")
    
    # Additional arguments for model loading with memory optimizations
    model_kwargs = {
        "quantization_config": quant_config,
        "torch_dtype": compute_dtype,
        "trust_remote_code": model_args.trust_remote_code,
    }
    
    # Add use_cache parameter conditionally - Gemma models don't accept this parameter
    # Print the model name for debugging
    logger.info(f"Model name: {model_args.model_name_or_path}")
    if "gemma" not in model_args.model_name_or_path.lower():
        logger.info("Adding use_cache parameter for non-Gemma model")
        model_kwargs["use_cache"] = False if training_args.gradient_checkpointing else True
    else:
        logger.info("Skipping use_cache parameter for Gemma model")
        
    # Add device_map for optimal memory usage on multi-GPU setup
    # For DeepSpeed, we don't need to specify the device_map as the model will be 
    # distributed across GPUs by DeepSpeed
    if not training_args.deepspeed:
        logger.info("Setting device map for non-DeepSpeed training")
        if torch.cuda.device_count() > 1 and not deepspeed_fallback:
            # For multiple GPUs, use device_map="auto"
            model_kwargs["device_map"] = "auto"
        else:
            # For single GPU, place everything on cuda:0
            model_kwargs["device_map"] = {"": 0}
    else:
        logger.info("Using DeepSpeed for distributed training - device_map will be handled by DeepSpeed")
    
    # Log system information
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    
    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path} with kwargs: {model_kwargs}")
    
    # Check for and fix missing vocab_size in Gemma-3 config
    if "gemma" in model_args.model_name_or_path.lower():
        try:
            # First load config to check for issues
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code
            )
            
            # For Gemma-3 models, the architecture attributes are in text_config but the model
            # implementation looks for them at the top level
            config_modified = False
            
            # Fix for nested text_config in Gemma-3
            if hasattr(config, 'text_config'):
                logger.info("Found nested text_config in Gemma config - copying attributes to top level")
                text_config = config.text_config
                
                # Copy all attributes from text_config to the main config
                for attr_name in dir(text_config):
                    # Skip private attributes and methods
                    if not attr_name.startswith('_') and not callable(getattr(text_config, attr_name)):
                        if not hasattr(config, attr_name):
                            value = getattr(text_config, attr_name)
                            logger.info(f"Copying {attr_name}={value} from text_config to main config")
                            setattr(config, attr_name, value)
                            config_modified = True
            
            # Ensure vocab_size is set
            if not hasattr(config, 'vocab_size'):
                logger.warning("Gemma config is missing vocab_size attribute - adding it from tokenizer")
                vocab_size = len(tokenizer.get_vocab())
                logger.info(f"Setting vocab_size to {vocab_size} from tokenizer vocabulary")
                config.vocab_size = vocab_size
                config_modified = True
            
            if config_modified:
                model_kwargs["config"] = config
                logger.info("Using modified config with added/copied attributes")
                
        except Exception as e:
            logger.warning(f"Error while checking/fixing config: {e}")
            logger.exception("Detailed error:")
    
    # Load the model with our potentially modified config
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    
    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Prepare model for training if using quantization
    if model_args.load_in_8bit or model_args.load_in_4bit:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )
    
    # Select target modules based on model type
    if "gemma" in model_args.model_name_or_path.lower():
        target_modules = lora_args.target_modules_for_gemma
    else:
        target_modules = lora_args.target_modules_for_llama
    
    logger.info(f"Using LoRA target modules: {target_modules}")
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    # Log model parameter info
    logger.info(f"Trainable params: {model.print_trainable_parameters()}")
    
    # Load dataset
    train_dataset = JsonlDataset(
        dataset_path=data_args.dataset_path,
        tokenizer=tokenizer,
        template_yaml=data_args.template_path,
        max_seq_length=data_args.max_seq_length,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    if training_args.resume_from_checkpoint:
        checkpoint = training_args.resume_from_checkpoint
    else:
        checkpoint = get_last_checkpoint(training_args.output_dir)
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save model
    trainer.save_model()
    trainer.save_state()
    
    # Log and save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 