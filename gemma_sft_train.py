#!/usr/bin/env python
# coding=utf-8

import os
import yaml
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any, Union

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
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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
    enable_debug: bool = field(
        default=False,
        metadata={"help": "Enable additional debugging output"}
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

@dataclass
class DataCollatorForCausalLM:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "pt", "tf", "np", "jax".
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    return_tensors: str = "pt"
    label_pad_token_id: int = -100

    def __call__(self, features):
        # First, ensure sequences are not too long
        if self.max_length is not None:
            for feature in features:
                if len(feature['input_ids']) > self.max_length:
                    # Truncate to max_length
                    feature['input_ids'] = feature['input_ids'][:self.max_length]
                    feature['attention_mask'] = feature['attention_mask'][:self.max_length]
                    if 'labels' in feature:
                        feature['labels'] = feature['labels'][:self.max_length]
        
        # Save original labels
        if "labels" in features[0]:
            labels = [feature.pop("labels") for feature in features]
        else:
            labels = None
            
        # Pad the inputs (input_ids and attention_mask)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Add labels back with proper padding
        if labels is not None:
            # Pad the labels (replacing pad with label_pad_token_id)
            padded_labels = self.tokenizer.pad(
                {"input_ids": labels},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )["input_ids"]
            
            # Replace padding token with label_pad_token_id (-100)
            if self.tokenizer.pad_token_id != self.label_pad_token_id:
                padded_labels[padded_labels == self.tokenizer.pad_token_id] = self.label_pad_token_id
                
            # Add labels to batch
            batch["labels"] = padded_labels
            
        return batch

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
            
            # Add BOS token if not already present
            if len(text_parts) == 0 or text_parts[0] != self.tokenizer.bos_token_id:
                text_parts.insert(0, self.tokenizer.bos_token_id)
                loss_mask_parts.insert(0, 0.0)  # No loss on BOS token
            
            # Add EOS token
            text_parts.append(self.tokenizer.eos_token_id)
            loss_mask_parts.append(1.0)  # Apply loss on EOS token
            
            # Truncate if exceeding max_length
            if len(text_parts) > self.max_seq_length:
                logger.warning(f"Example {idx} exceeds max_seq_length ({len(text_parts)} > {self.max_seq_length}), truncating")
                text_parts = text_parts[:self.max_seq_length]
                loss_mask_parts = loss_mask_parts[:self.max_seq_length]
            
            # Convert to list format that can be properly padded by data collator
            input_ids = text_parts
            attention_mask = [1] * len(input_ids)
            labels = input_ids.copy()
            
            # Apply loss mask - set tokens with mask=0.0 to -100 in labels
            for i, mask_val in enumerate(loss_mask_parts):
                if mask_val == 0.0:
                    labels[i] = -100
            
            # Debug info about label distribution
            if hasattr(self, 'enable_debug') and self.enable_debug and idx < 3:
                non_masked = sum(1 for l in labels if l != -100)
                masked = sum(1 for l in labels if l == -100)
                logger.info(f"Debug __getitem__ idx={idx} using template:")
                logger.info(f"  Total length: {len(input_ids)} tokens")
                logger.info(f"  Number of -100 labels: {masked}")
                logger.info(f"  Number of non-masked labels: {non_masked}")
                if non_masked == 0:
                    logger.error("  ❌ ERROR: All labels are masked! Model will not learn anything.")
                
            # Return as simple lists instead of tensors to allow proper padding by collator
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
            
            # Use tokenizer with explicit padding and truncation settings
            encodings = self.tokenizer(
                text, 
                max_length=self.max_seq_length,
                truncation=True, 
                padding=False, # Let the data collator handle padding
                return_tensors=None # Return lists, not tensors
            )
            
            # Verify lengths are within max_seq_length
            if len(encodings["input_ids"]) > self.max_seq_length:
                logger.warning(f"Example {idx} exceeds max_seq_length after tokenization ({len(encodings['input_ids'])} > {self.max_seq_length}), truncating")
                encodings["input_ids"] = encodings["input_ids"][:self.max_seq_length]
                encodings["attention_mask"] = encodings["attention_mask"][:self.max_seq_length]
            
            # Create labels: -100 for prompt tokens (no loss), actual token IDs for output tokens
            prompt_len = min(len(self.tokenizer(prompt, truncation=True).input_ids), len(encodings["input_ids"]))
            labels = encodings["input_ids"].copy()
            labels[:prompt_len] = [-100] * prompt_len  # No loss for prompt tokens
            
            # Debug info about label distribution
            if hasattr(self, 'enable_debug') and self.enable_debug and idx < 3:
                non_masked = len(labels) - labels.count(-100)
                logger.info(f"Debug __getitem__ idx={idx} using default format:")
                logger.info(f"  Prompt: {prompt[:50]}...")
                logger.info(f"  Prompt length: {prompt_len} tokens")
                logger.info(f"  Total length: {len(encodings['input_ids'])} tokens")
                logger.info(f"  Number of -100 labels: {labels.count(-100)}")
                logger.info(f"  Number of non-masked labels: {non_masked}")
                if non_masked == 0:
                    logger.error("  ❌ ERROR: All labels are masked! Model will not learn anything.")
                elif prompt_len == len(encodings["input_ids"]):
                    logger.error("  ❌ ERROR: Prompt length equals total length! No tokens left for prediction.")
            
            return {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels
            }

# Add DebugCallback for monitoring training progress
class DebugCallback(TrainerCallback):
    """Custom callback for debugging training issues."""
    
    def __init__(self):
        self.last_loss = None
        self.loss_counter = 0
        self.static_loss_count = 0
        self.step_counter = 0
        self.detailed_logging = True  # Always do detailed logging to diagnose issues
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Monitor loss changes and gradients during training."""
        self.step_counter += 1
        
        if state.log_history and len(state.log_history) > 0:
            # Extract the current loss
            current_loss = None
            for entry in reversed(state.log_history):
                if 'loss' in entry:
                    current_loss = entry['loss']
                    break
            
            if current_loss is not None:
                # Check if loss is changing
                if self.last_loss is not None and abs(current_loss - self.last_loss) < 1e-6:
                    self.static_loss_count += 1
                    if self.static_loss_count >= 5:
                        logger.warning(f"⚠️ Loss has been static at {current_loss:.6f} for {self.static_loss_count} steps!")
                        
                        # If loss is static, do more detailed debugging
                        if model is not None and self.detailed_logging:
                            logger.warning("Performing detailed gradient inspection due to static loss:")
                            self._inspect_model_and_gradients(model)
                else:
                    if self.last_loss is not None:
                        logger.info(f"Loss changed from {self.last_loss:.6f} to {current_loss:.6f} (delta: {current_loss - self.last_loss:.6f})")
                    self.static_loss_count = 0
                
                self.last_loss = current_loss
        
        # Check gradients regularly
        if model is not None and (self.step_counter <= 10 or self.step_counter % 10 == 0):
            self._check_gradients(model)
    
    def _check_gradients(self, model):
        """Check gradients in LoRA layers."""
        has_grad = False
        zero_grad_count = 0
        non_zero_grad_count = 0
        lora_layer_count = 0
        
        # Track maximum gradient norm for reporting
        max_grad_norm = 0.0
        max_grad_name = ""
        
        # First check if we can find any lora parameters at all
        lora_params = [name for name, param in model.named_parameters() 
                      if param.requires_grad and ('lora_' in name or 'adapter' in name)]
        
        if not lora_params:
            logger.error("❌ NO LORA PARAMETERS FOUND! Model appears not to have LoRA layers properly initialized")
            # List some of the actual trainable parameters to help diagnose
            trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
            if trainable_params:
                logger.error(f"Found {len(trainable_params)} trainable parameters, but none are LoRA parameters")
                logger.error(f"First few trainable parameters: {trainable_params[:5]}")
            else:
                logger.error("No trainable parameters found at all!")
            return
            
        logger.info(f"Found {len(lora_params)} LoRA trainable parameters to check for gradients")
        
        # Now check gradients for those parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'lora_' in name or 'adapter' in name:
                    lora_layer_count += 1
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm().item()
                        
                        if grad_norm > 0:
                            has_grad = True
                            non_zero_grad_count += 1
                            
                            # Track maximum gradient
                            if grad_norm > max_grad_norm:
                                max_grad_norm = grad_norm
                                max_grad_name = name
                        else:
                            zero_grad_count += 1
                        
                        # Log a sample of gradient norms
                        if lora_layer_count <= 3 or grad_norm > 0:  # Log first few layers and any non-zero gradients
                            logger.info(f"Step {self.step_counter} - Grad norm for {name}: {grad_norm:.6f}")
        
        # Show summary stats for this step
        grad_status = "✅ GRADIENTS DETECTED" if has_grad else "❌ NO GRADIENTS"
        logger.info(f"Step {self.step_counter}: {grad_status} - LoRA layers with zero gradients: {zero_grad_count}/{lora_layer_count}")
        
        if has_grad:
            logger.info(f"Step {self.step_counter}: Max gradient: {max_grad_norm:.6f} in {max_grad_name}")
        
        if not has_grad and lora_layer_count > 0:
            logger.warning("❌ NO GRADIENT FLOW DETECTED! All trainable parameters have zero gradients.")
    
    def _inspect_model_and_gradients(self, model):
        """Do a deep inspection of the model and its gradients."""
        logger.warning("=== DETAILED MODEL INSPECTION ===")
        
        # Check if model is in training mode
        logger.warning(f"Model training mode: {model.training}")
        
        # Try to detect if we're using DeepSpeed and inspect its state
        try:
            if hasattr(model, 'optimizer') and hasattr(model.optimizer, 'loss_scale'):
                logger.warning(f"Loss scale: {model.optimizer.loss_scale}")
                if hasattr(model.optimizer, 'cur_scale'):
                    logger.warning(f"Current scale: {model.optimizer.cur_scale}")
                if hasattr(model.optimizer, 'inf_has_occurred'):
                    logger.warning(f"Infinity detected: {model.optimizer.inf_has_occurred}")
        except Exception as e:
            logger.warning(f"Error inspecting optimizer: {e}")
        
        # Log extreme parameter values that might cause gradient issues
        try:
            for name, param in model.named_parameters():
                if param.requires_grad and 'lora_' in name:
                    # Check for NaN or extreme values in parameters
                    has_nan = torch.isnan(param.data).any().item()
                    has_inf = torch.isinf(param.data).any().item()
                    max_val = param.data.abs().max().item()
                    
                    if has_nan or has_inf or max_val > 1000:
                        logger.warning(f"Parameter {name}: NaN={has_nan}, Inf={has_inf}, Max abs value={max_val}")
                    
                    # Check for NaN or extreme values in gradients if they exist
                    if param.grad is not None:
                        grad_has_nan = torch.isnan(param.grad.data).any().item()
                        grad_has_inf = torch.isinf(param.grad.data).any().item()
                        grad_max_val = param.grad.data.abs().max().item() if param.grad.data.numel() > 0 else 0
                        
                        if grad_has_nan or grad_has_inf:
                            logger.warning(f"Gradient for {name}: NaN={grad_has_nan}, Inf={grad_has_inf}, Max abs value={grad_max_val}")
        except Exception as e:
            logger.warning(f"Error inspecting parameters: {e}")
        
        logger.warning("=== END DETAILED INSPECTION ===")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of training."""
        if model is not None:
            logger.info("Initial model check at start of training:")
            self._check_gradients(model)
            
            # Add hook to track gradient norms
            def gradient_hook(grad):
                if grad is not None:
                    grad_norm = grad.norm().item()
                    if grad_norm == 0:
                        logger.info(f"Zero gradient detected in hook during backward pass")
                    return grad
            
            # Add hooks to a sample of parameters
            hook_count = 0
            for name, param in model.named_parameters():
                if param.requires_grad and 'lora_' in name and hook_count < 5:
                    param.register_hook(gradient_hook)
                    hook_count += 1
                    logger.info(f"Added gradient hook to {name}")
        
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of training."""
        logger.info(f"Training completed after {self.step_counter} steps")
        if model is not None:
            self._check_gradients(model)

# Add new ModelDiagnostics class for deep model inspection
class ModelDiagnostics:
    """Diagnostic tool to deeply analyze model structure, trainable parameters, and computation graph."""
    
    @staticmethod
    def analyze_model(model, tokenizer=None, detailed=True):
        """Perform comprehensive model analysis and print results."""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE MODEL DIAGNOSTICS")
        logger.info("=" * 80)
        
        # Basic model information
        logger.info("MODEL CLASS: " + str(type(model)))
        logger.info("MODEL MODE: " + ("TRAINING" if model.training else "EVALUATION"))
        
        # Check if model is wrapped by DeepSpeed, FSDP, etc.
        ModelDiagnostics._check_model_wrappers(model)
        
        # Analyze trainable parameters in detail
        params_by_module = ModelDiagnostics._analyze_trainable_parameters(model)
        
        # Check if PEFT/LoRA is properly applied
        ModelDiagnostics._check_peft_application(model)
        
        # Analyze computation graph if detailed analysis requested
        if detailed:
            ModelDiagnostics._analyze_computation_graph(model, tokenizer)
        
        logger.info("=" * 80)
        logger.info("COMPLETED MODEL DIAGNOSTICS")
        logger.info("=" * 80)
        
        return params_by_module
    
    @staticmethod
    def _check_model_wrappers(model):
        """Detect if model is wrapped by training frameworks."""
        logger.info("\nCHECKING FOR MODEL WRAPPERS:")
        
        # Check common wrapper patterns
        wrapper_detected = False
        
        # Check DeepSpeed wrapper
        if hasattr(model, 'module') and 'deepspeed' in str(type(model)).lower():
            logger.info("✅ Model is wrapped by DeepSpeed")
            wrapper_detected = True
            # Log DeepSpeed config if available
            if hasattr(model, 'config'):
                config = model.config
                logger.info("DeepSpeed Zero Stage: " + str(config.get('zero_optimization', {}).get('stage', 'Not found')))
        
        # Check for FSDP wrapper
        if 'FSDP' in str(type(model)):
            logger.info("✅ Model is wrapped by FSDP (Fully Sharded Data Parallel)")
            wrapper_detected = True
        
        # Check for DDP wrapper
        if hasattr(model, 'module') and 'DistributedDataParallel' in str(type(model)):
            logger.info("✅ Model is wrapped by PyTorch DistributedDataParallel")
            wrapper_detected = True
        
        # Check for PEFT wrapper
        if 'PeftModel' in str(type(model)):
            logger.info("✅ Model is wrapped by PEFT (Parameter-Efficient Fine-Tuning)")
            wrapper_detected = True
            # Additional PEFT-specific checks
            if hasattr(model, 'base_model'):
                logger.info("   - Has base_model attribute ✓")
                
                # Check peft_config
                if hasattr(model, 'peft_config'):
                    logger.info("   - Has peft_config ✓")
                    # Print key peft_config details if available
                    peft_config = model.peft_config
                    if isinstance(peft_config, dict):
                        for adapter_name, config in peft_config.items():
                            logger.info(f"   - Adapter: {adapter_name}")
                            if hasattr(config, 'target_modules'):
                                logger.info(f"     - Target modules: {config.target_modules}")
                            if hasattr(config, 'lora_alpha'):
                                logger.info(f"     - LoRA alpha: {config.lora_alpha}")
                            if hasattr(config, 'lora_dropout'):
                                logger.info(f"     - LoRA dropout: {config.lora_dropout}")
        
        if not wrapper_detected:
            logger.info("❌ No common model wrappers detected. Using plain PyTorch model.")
    
    @staticmethod
    def _analyze_trainable_parameters(model):
        """Analyze trainable parameters and their distribution across modules."""
        logger.info("\nANALYZING TRAINABLE PARAMETERS:")
        
        # Count parameters by module prefix
        params_by_module = {}
        total_params = 0
        trainable_params = 0
        
        # Dictionary to track parameter sizes per module
        module_sizes = {}
        
        for name, param in model.named_parameters():
            # Count all parameters
            param_size = param.numel()
            total_params += param_size
            
            # Process module name to get module prefix
            module_path = name.split('.')
            for i in range(len(module_path)):
                prefix = '.'.join(module_path[:i+1])
                if prefix not in module_sizes:
                    module_sizes[prefix] = 0
                module_sizes[prefix] += param_size
            
            # Track trainable parameters
            if param.requires_grad:
                trainable_params += param_size
                
                # Extract module prefix (first parts of parameter name)
                parts = name.split('.')
                module_prefix = parts[0]
                if len(parts) > 2:
                    module_prefix = f"{parts[0]}.{parts[1]}"
                
                # Add to module tracking
                if module_prefix not in params_by_module:
                    params_by_module[module_prefix] = {
                        'count': 0, 
                        'size': 0,
                        'names': []
                    }
                
                params_by_module[module_prefix]['count'] += 1
                params_by_module[module_prefix]['size'] += param_size
                
                # Add parameter name to list (limit to first 5 for readability)
                if len(params_by_module[module_prefix]['names']) < 5:
                    params_by_module[module_prefix]['names'].append(name)
        
        # Print overall statistics
        trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
        
        # Sort modules by parameter count and print top 10
        sorted_modules = sorted(params_by_module.items(), key=lambda x: x[1]['size'], reverse=True)
        logger.info("\nTRAINABLE PARAMETERS BY MODULE (TOP 10):")
        
        for i, (module_name, data) in enumerate(sorted_modules[:10]):
            module_percent = 100 * data['size'] / trainable_params if trainable_params > 0 else 0
            logger.info(f"#{i+1}: {module_name} - {data['count']} params, {data['size']:,} elements ({module_percent:.2f}% of trainable)")
            logger.info(f"     Example parameters: {', '.join(data['names'][:3])}")
        
        # Check for suspicious modules that should typically be frozen
        logger.info("\nCHECKING FOR SUSPICIOUS TRAINABLE MODULES:")
        suspicious_detected = False
        suspicious_prefixes = [
            'embeddings', 'embed_tokens', 'norm', 'ln_', 'layernorm'
        ]
        
        for module_prefix in params_by_module.keys():
            for suspicious in suspicious_prefixes:
                if suspicious.lower() in module_prefix.lower():
                    logger.info(f"⚠️ Potentially suspicious trainable module: {module_prefix} with {params_by_module[module_prefix]['size']:,} elements")
                    logger.info(f"     Example parameters: {', '.join(params_by_module[module_prefix]['names'][:3])}")
                    suspicious_detected = True
        
        if not suspicious_detected:
            logger.info("✅ No suspicious trainable modules detected.")
        
        # Look for largest top-level modules (useful to identify where most parameters are)
        logger.info("\nLARGEST MODEL COMPONENTS:")
        top_level_modules = {}
        
        for path, size in module_sizes.items():
            top_level = path.split('.')[0]
            if top_level not in top_level_modules:
                top_level_modules[top_level] = 0
            top_level_modules[top_level] += size
        
        sorted_top_modules = sorted(top_level_modules.items(), key=lambda x: x[1], reverse=True)
        for name, size in sorted_top_modules[:5]:
            percent = 100 * size / total_params
            logger.info(f"{name}: {size:,} elements ({percent:.2f}% of model)")
        
        return params_by_module
    
    @staticmethod
    def _check_peft_application(model):
        """Check if PEFT/LoRA adapters are properly applied to the model."""
        logger.info("\nCHECKING PEFT/LORA APPLICATION:")
        
        # Check if model has expected PEFT attributes
        has_peft_structure = False
        
        if hasattr(model, 'peft_config'):
            has_peft_structure = True
            logger.info("✅ Model has peft_config attribute")
        
        # Look for LoRA modules in model structure
        lora_modules_found = []
        
        def find_lora_modules(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if this is a LoRA module
                if 'lora_' in name:
                    lora_modules_found.append(full_name)
                
                # Continue recursion
                find_lora_modules(child, full_name)
        
        # Start recursive search
        find_lora_modules(model)
        
        if lora_modules_found:
            logger.info(f"✅ Found {len(lora_modules_found)} LoRA modules in model")
            # Print first few LoRA modules
            for module in lora_modules_found[:5]:
                logger.info(f"   - {module}")
        else:
            logger.info("❌ No LoRA modules found in model structure")
        
        # Check for active hooks in forward methods (particularly important to verify LoRA integration)
        hooks_detected = False
        
        # Helper function to find modules with hooks
        def check_for_hooks(module, prefix=''):
            nonlocal hooks_detected
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check for forward hooks
                if hasattr(child, '_forward_hooks') and child._forward_hooks:
                    hooks_detected = True
                    logger.info(f"✅ Found forward hooks on module: {full_name}")
                    logger.info(f"   - Hook count: {len(child._forward_hooks)}")
                
                # Check for forward pre-hooks
                if hasattr(child, '_forward_pre_hooks') and child._forward_pre_hooks:
                    hooks_detected = True
                    logger.info(f"✅ Found forward pre-hooks on module: {full_name}")
                    logger.info(f"   - Pre-hook count: {len(child._forward_pre_hooks)}")
                
                # Continue recursion
                check_for_hooks(child, full_name)
        
        # Start recursive search for hooks
        check_for_hooks(model)
        
        if not hooks_detected:
            logger.info("❌ No forward hooks detected in model. LoRA may not be properly integrated.")
            
            if has_peft_structure and lora_modules_found:
                logger.info("⚠️ Model has LoRA modules but no hooks - this suggests incomplete initialization")
    
    @staticmethod
    def _analyze_computation_graph(model, tokenizer=None):
        """Analyze computation graph to check if LoRA modules are connected."""
        logger.info("\nANALYZING COMPUTATION GRAPH CONNECTION:")
        
        # If no tokenizer provided, we can only do limited analysis
        if tokenizer is None:
            logger.info("ℹ️ Tokenizer not provided, skipping input-based graph analysis")
            return
        
        # Create a simple input to trace through the model
        try:
            logger.info("Generating sample input and tracing computation...")
            
            # Create a small sample input
            sample_text = "This is a test input to check model connections."
            inputs = tokenizer(sample_text, return_tensors="pt")
            
            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Set up hooks to track LoRA module activations
            activation_count = 0
            
            def hook_fn(module, input, output):
                nonlocal activation_count
                activation_count += 1
                return output
            
            # Register hooks on LoRA modules
            hooks = []
            for name, module in model.named_modules():
                if 'lora_' in name:
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
            
            # Run a forward pass with no grad to test connections
            logger.info("Running test forward pass...")
            with torch.no_grad():
                _ = model(**inputs)
            
            # Report activation results
            if activation_count > 0:
                logger.info(f"✅ LoRA modules were activated {activation_count} times during forward pass")
            else:
                logger.info("❌ NO LORA ACTIVATIONS DETECTED during forward pass")
                logger.info("   This confirms LoRA modules are not connected to computation graph")
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
        except Exception as e:
            logger.warning(f"Error during computation graph analysis: {e}")
            logger.warning("This may indicate issues with the model's forward pass")

# Add a function to analyze optimizer and scheduler configuration
def analyze_optimizer_and_scheduler(trainer):
    """Analyze optimizer and learning rate scheduler configuration."""
    logger.info("=" * 80)
    logger.info("ANALYZING OPTIMIZER AND SCHEDULER")
    logger.info("=" * 80)
    
    # Check optimizer
    if hasattr(trainer, 'optimizer') and trainer.optimizer:
        optimizer = trainer.optimizer
        logger.info(f"Optimizer type: {type(optimizer)}")
        
        # Check parameter groups
        if hasattr(optimizer, 'param_groups'):
            logger.info(f"Number of parameter groups: {len(optimizer.param_groups)}")
            
            # Check first few parameter groups
            for i, group in enumerate(optimizer.param_groups[:3]):
                logger.info(f"Group {i}:")
                logger.info(f"  Learning rate: {group.get('lr', 'N/A')}")
                logger.info(f"  Weight decay: {group.get('weight_decay', 'N/A')}")
                logger.info(f"  Parameter count: {len(group.get('params', []))}")
                
                # Sample some parameters from this group
                params = group.get('params', [])
                if params:
                    # Try to find parameter names for these params
                    for param in params[:2]:
                        # Report parameter details
                        if hasattr(param, 'shape'):
                            logger.info(f"  Parameter shape: {param.shape}")
                        if hasattr(param, 'requires_grad'):
                            logger.info(f"  requires_grad: {param.requires_grad}")
    else:
        logger.info("❌ No optimizer found in trainer")
    
    # Check learning rate scheduler
    if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler:
        scheduler = trainer.lr_scheduler
        logger.info(f"Learning rate scheduler type: {type(scheduler)}")
        
        # Check if it's a known scheduler type
        scheduler_type = str(type(scheduler))
        if 'get_linear_schedule' in scheduler_type:
            logger.info("✅ Using linear learning rate schedule")
        elif 'get_cosine_schedule' in scheduler_type:
            logger.info("✅ Using cosine learning rate schedule")
        elif 'get_constant_schedule' in scheduler_type:
            logger.info("❌ Using constant learning rate schedule - may not be optimal for training")
        
        # Try to extract current learning rate
        if hasattr(scheduler, 'get_last_lr'):
            logger.info(f"Current learning rate: {scheduler.get_last_lr()}")
    else:
        logger.info("No learning rate scheduler found in trainer")
    
    logger.info("=" * 80)

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
    
    # Add debug option
    debug_mode = model_args.enable_debug
    if debug_mode:
        logger.info("Debug mode enabled - will output detailed debugging information")
    
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
        # For Gemma-3 models, use eager attention implementation as recommended
        logger.info("Setting attn_implementation='eager' for Gemma model")
        model_kwargs["attn_implementation"] = "eager"
        
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
    
    # Instead of trying to replace the embedding layer, let's properly handle Gemma's
    # initialization by forcing a tokenization and forward pass before training
    if "gemma" in model_args.model_name_or_path.lower():
        logger.info("Gemma model detected - initializing model weights properly...")
        
        # Log the current embedding layer state if it exists
        if hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'weight'):
            embed_shape = model.model.embed_tokens.weight.shape
            logger.info(f"Initial embed_tokens weight shape: {embed_shape}")
            
            # If it appears to be improperly initialized, force initialization through a test forward pass
            if embed_shape[0] == 0 or (len(embed_shape) == 1):
                logger.info("Embedding appears uninitialized. Performing test forward pass to properly initialize...")
                
                # Create a small test input
                test_input = tokenizer("Hello, world!", return_tensors="pt")
                
                # Move to appropriate device
                if hasattr(model, 'device'):
                    test_input = {k: v.to(model.device) for k, v in test_input.items()}
                
                # Run a test forward pass with no grad to initialize layers
                with torch.no_grad():
                    try:
                        # Don't care about the output, just need to run the forward pass
                        _ = model(**test_input)
                        
                        # Now check if the embedding was properly initialized
                        if hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'weight'):
                            new_shape = model.model.embed_tokens.weight.shape
                            logger.info(f"After test forward pass, embed_tokens shape: {new_shape}")
                            
                            if new_shape[0] == 0:
                                logger.warning("Embedding still has zero size after forward pass")
                                # Fall back to config-based initialization only if necessary
                                vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else len(tokenizer)
                                hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 5376
                                logger.warning(f"Will create embedding with vocab_size={vocab_size}, hidden_size={hidden_size}")
                    
                    except Exception as e:
                        logger.warning(f"Test forward pass failed: {e}")
                        logger.info("Will proceed with regular training and let the model handle initialization")
    
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
    
    # CRITICAL: Properly freeze base model parameters BEFORE applying LoRA
    logger.info("Completely freezing base model parameters before LoRA application...")
    
    # First pass: Enforce module-level freezing  
    for name, module in model.named_modules():
        if hasattr(module, 'requires_grad'):
            module.requires_grad = False
    
    # Second pass: Ensure all individual parameters are frozen
    trainable_before_lora = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_before_lora.append(name)
        param.requires_grad = False
    
    if trainable_before_lora:
        logger.warning(f"Found {len(trainable_before_lora)} trainable parameters before LoRA application (now frozen)")
        logger.warning(f"Examples: {trainable_before_lora[:5]}")
    else:
        logger.info("✅ All base model parameters correctly frozen before LoRA application")
    
    # Verify actual parameter state before LoRA
    still_trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if still_trainable:
        logger.error(f"❌ Still have {len(still_trainable)} trainable parameters after freezing attempt!")
        logger.error(f"Examples: {still_trainable[:5]}")
        raise ValueError("Failed to completely freeze model before LoRA application - this will cause problems")
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=None,  # Don't save any modules outside of LoRA parameters
        bias="none",  # Don't train bias parameters
        fan_in_fan_out=False,  # For PyTorch Linear weights that have shape [in, out], set to True
    )
    
    # Apply LoRA to model
    logger.info("Applying LoRA to model...")
    
    # Explicitly ensure all base parameters are frozen before adding LoRA
    for param in model.parameters():
        param.requires_grad = False
        
    model = get_peft_model(model, peft_config)
    
    # Add a thorough verification function to repair LoRA issues
    def verify_and_repair_lora_modules(model):
        """Deeply inspect and repair LoRA integration issues."""
        logger.info("=" * 80)
        logger.info("VERIFYING AND REPAIRING LORA MODULES")
        logger.info("=" * 80)
        
        # Check embedding layer status
        logger.info("Checking embedding layer status...")
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            base = model.base_model.model
            if hasattr(base, 'embed_tokens') and hasattr(base.embed_tokens, 'weight'):
                embed_shape = base.embed_tokens.weight.shape
                logger.info(f"Embedding layer shape in base model: {embed_shape}")
                
                if embed_shape[0] == 0:
                    logger.warning("⚠️ Base model has EMPTY embedding layer - this will cause forward pass failures")
                    logger.warning("Embedding issues should have been fixed before LoRA application")
                    logger.warning("Attempting to diagnose embedding references...")
                    
                    # Check alternative paths to embedding layer
                    if hasattr(model, 'get_input_embeddings'):
                        embed = model.get_input_embeddings()
                        if embed is not None and hasattr(embed, 'weight'):
                            logger.info(f"Model's get_input_embeddings() shape: {embed.weight.shape}")
                    
                    # No direct fix here - we're just diagnosing
        
        # 1. Check for any module requiring grad whose name doesn't contain lora/adapter
        # First collect all parameters and modules so we don't modify during iteration
        non_lora_trainable_params = []
        lora_modules = []
        regular_target_modules = []
        
        # Collect all modules for inspection
        for name, module in model.named_modules():
            if any(target in name for target in target_modules):
                # Found a target module (should have a LoRA module attached)
                regular_target_modules.append(name)
            
            # Find any LoRA modules
            if 'lora_' in name:
                lora_modules.append(name)
                
        # Log found modules
        logger.info(f"Found {len(lora_modules)} LoRA modules")
        logger.info(f"Found {len(regular_target_modules)} potential target modules (containing target_modules keywords)")
        
        # Check for any parameters requiring grad that aren't LoRA
        for name, param in model.named_parameters():
            if param.requires_grad and not any(lora_name in name for lora_name in ['lora_', 'adapter']):
                non_lora_trainable_params.append(name)
        
        if non_lora_trainable_params:
            logger.warning(f"Found {len(non_lora_trainable_params)} non-LoRA parameters with requires_grad=True")
            logger.warning(f"First few: {non_lora_trainable_params[:5]}")
            logger.warning("Forcefully freezing these parameters...")
            
            for name, param in model.named_parameters():
                if name in non_lora_trainable_params:
                    param.requires_grad = False
            
            # Verify freezing worked
            still_trainable = []
            for name, param in model.named_parameters():
                if param.requires_grad and not any(lora_name in name for lora_name in ['lora_', 'adapter']):
                    still_trainable.append(name)
            
            if still_trainable:
                logger.error(f"Failed to freeze {len(still_trainable)} parameters")
            else:
                logger.info("Successfully froze all non-LoRA parameters")
        
        # 2. Check for proper connections and hooks
        # Check if proper hooks are registered for linear layers with LoRA
        import torch.nn as nn
        hooks_found = False
        unhookeLinears = []
        
        for name, module in model.named_modules():
            # Find linear modules that should have hooks but don't
            if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
                # Check for forward hooks
                if (not hasattr(module, '_forward_hooks') or not module._forward_hooks or
                    not hasattr(module, '_forward_pre_hooks') or not module._forward_pre_hooks):
                    unhookeLinears.append(name)
        
        if unhookeLinears:
            logger.warning(f"Found {len(unhookeLinears)} linear modules missing hooks!")
            logger.warning(f"Examples: {unhookeLinears[:5]}")
            logger.warning("This suggests a problem with LoRA initialization.")
            logger.warning("Attempting to repair by reinitializing PEFT...")
            
            # Strategy: Create LoRA configuration again and reapply
            # First let's clean up any broken partial LoRA state
            # We need to restore the original modules to start
            
            # Step 1: Collect existing LoRA A/B matrices to preserve any initialization
            lora_state = {}
            for name, module in model.named_modules():
                if 'lora_A' in name or 'lora_B' in name:
                    # Store the state
                    if hasattr(module, 'weight'):
                        # Get the full path with parameters
                        weight_name = f"{name}.weight"
                        lora_state[weight_name] = module.weight.detach().clone()
            
            logger.info(f"Preserved {len(lora_state)} existing LoRA weights")
            
            # Step 2: Reapply LoRA
            try:
                logger.info("Attempting to re-apply PEFT LoRA transformation...")
                from peft import get_peft_model_state_dict
                
                # First ensure base model parameters are all frozen
                for param in model.parameters():
                    param.requires_grad = False
                
                # Get base model (unwrapped from PEFT)
                if hasattr(model, 'base_model'):
                    logger.info("Using model.base_model for re-applying LoRA")
                    base_model = model.base_model
                    
                    # Reapply LoRA configuration
                    from peft import get_peft_model
                    new_model = get_peft_model(base_model, peft_config)
                    
                    logger.info("Successfully reapplied LoRA transformation")
                    
                    # Update the current model reference to the new model
                    model = new_model
                    
                    # Optional: restore saved LoRA weights if needed
                    if lora_state:
                        logger.info("Restoring saved LoRA weights...")
                        mismatch = 0
                        restored = 0
                        
                        # Copy saved weights back into new LoRA modules
                        for name, param in model.named_parameters():
                            if name in lora_state:
                                if param.shape == lora_state[name].shape:
                                    param.data.copy_(lora_state[name])
                                    restored += 1
                                else:
                                    logger.warning(f"Shape mismatch for {name}: {param.shape} vs {lora_state[name].shape}")
                                    mismatch += 1
                        
                        logger.info(f"Restored {restored} LoRA weights, {mismatch} shape mismatches")
                else:
                    logger.error("Could not find model.base_model for re-applying LoRA")
            except Exception as e:
                logger.error(f"Error during LoRA reapplication: {e}")
                logger.exception("Detailed error:")
                
            # 3. Finally, verify LoRA parameters have requires_grad=True
            lora_trainable = [name for name, param in model.named_parameters() 
                             if param.requires_grad and any(lora_name in name for lora_name in ['lora_', 'adapter'])]
            
            if not lora_trainable:
                logger.error("No trainable LoRA parameters found after repair!")
                logger.error("Attempting one last fix...")
                
                # Force enable training only for LoRA parameters
                for name, param in model.named_parameters():
                    if any(lora_name in name for lora_name in ['lora_', 'adapter']):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                    
                # Verify again
                lora_trainable = [name for name, param in model.named_parameters() 
                                 if param.requires_grad and any(lora_name in name for lora_name in ['lora_', 'adapter'])]
                
                logger.info(f"After forced fix: {len(lora_trainable)} trainable LoRA parameters")
                
            logger.info("=" * 80)
            
            return model
        
    # Repair any LoRA module issues
    model = verify_and_repair_lora_modules(model)
    
    # Explicitly verify parameter freezing after LoRA
    base_params_trainable = []
    lora_params_trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_' in name or 'adapter' in name:
                lora_params_trainable.append(name)
            else:
                base_params_trainable.append(name)
    
    if base_params_trainable:
        logger.warning(f"WARNING: Found {len(base_params_trainable)} base model parameters still trainable after LoRA application!")
        logger.warning(f"First few: {base_params_trainable[:10]}")
        
        # Re-freeze any base parameters that might have been accidentally unfrozen
        logger.info("Re-freezing base model parameters...")
        for name, param in model.named_parameters():
            if not any(lora_name in name for lora_name in ['lora_', 'adapter']):
                param.requires_grad = False
                if param.requires_grad:
                    logger.error(f"FAILED to freeze parameter: {name}")
    
    # Check module-level requires_grad attributes (some models set these)
    module_requires_grad = []
    for name, module in model.named_modules():
        if hasattr(module, 'requires_grad') and module.requires_grad:
            module_requires_grad.append(name)
    
    if module_requires_grad:
        logger.warning(f"Found {len(module_requires_grad)} modules with requires_grad=True at module level")
        logger.warning(f"First few: {module_requires_grad[:5]}")
        
        # Try to fix module-level requires_grad as well
        for name, module in model.named_modules():
            if hasattr(module, 'requires_grad') and not any(lora_name in name for lora_name in ['lora_', 'adapter']):
                module.requires_grad = False
    
    # Perform a final verification of trainable parameters
    trainable_params_after_fix = [name for name, param in model.named_parameters() if param.requires_grad]
    logger.info(f"LoRA trainable parameters: {len(lora_params_trainable)}")
    logger.info(f"Trainable parameters after fixing: {len(trainable_params_after_fix)}")
    
    if len(trainable_params_after_fix) > len(lora_params_trainable):
        logger.warning("STILL HAVE NON-LORA TRAINABLE PARAMETERS!")
        non_lora = [p for p in trainable_params_after_fix if not any(lora_name in p for lora_name in ['lora_', 'adapter'])]
        logger.warning(f"Non-LoRA trainable parameters: {len(non_lora)}")
        logger.warning(f"Examples: {non_lora[:10]}")
        
        # Last resort: use model.requires_grad_ to force everything frozen
        model.requires_grad_(False)
        
        # Then selectively re-enable only LoRA parameters
        for name, param in model.named_parameters():
            if any(lora_name in name for lora_name in ['lora_', 'adapter']):
                param.requires_grad = True
        
        # Final verification
        final_trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        logger.info(f"Final trainable parameter count: {len(final_trainable)}")
    
    # Debug: Count trainable parameters and verify LoRA is properly applied
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")

    if trainable_params == 0:
        raise ValueError("❌ ERROR: No trainable parameters found! LoRA adapters might not be properly initialized.")

    # Verify target_modules were found in the model
    lora_targets_found = False
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            logger.info(f"✅ Found LoRA module at: {name}")
            lora_targets_found = True
            break

    if not lora_targets_found:
        raise ValueError("❌ ERROR: No LoRA modules found in model! Check if target_modules are correct.")
        
    # Test the model with a simple forward pass to ensure it's working before training
    logger.info("Testing the model with a simple forward pass...")
    test_input = tokenizer("Hello, world!", return_tensors="pt")
    
    # Move to the appropriate device
    device = next(model.parameters()).device
    test_input = {k: v.to(device) for k, v in test_input.items()}
    
    # Try a test forward pass with model in eval mode
    model_was_training = model.training
    model.eval()
    
    try:
        with torch.no_grad():
            _ = model(**test_input)
        logger.info("✅ Model forward pass succeeded")
        
        # Try to re-run without eval mode to test training mode
        if model_was_training:
            model.train()
            with torch.no_grad():
                _ = model(**test_input)
            logger.info("✅ Model forward pass also succeeds in training mode")
    except Exception as e:
        logger.error(f"❌ Model forward pass failed: {str(e)}")
        logger.error("The model may not be usable for training.")
        
        # Try to diagnose embedding issues
        if "embedding" in str(e).lower() or "size of tensor" in str(e).lower():
            logger.error("This appears to be an embedding initialization issue.")
            logger.error("Checking embedding references...")
            
            # Check main embedding
            if hasattr(model.base_model.model, 'embed_tokens'):
                embed = model.base_model.model.embed_tokens
                if hasattr(embed, 'weight'):
                    logger.error(f"Embedding shape: {embed.weight.shape}")
            
            # Check via API
            if hasattr(model, 'get_input_embeddings'):
                embed = model.get_input_embeddings()
                if embed is not None and hasattr(embed, 'weight'):
                    logger.error(f"get_input_embeddings() shape: {embed.weight.shape}")
    
    # Ensure model is back in training mode
    if model_was_training:
        model.train()

    # Run detailed diagnostics to understand model structure and parameter status
    logger.info("Running comprehensive model diagnostics...")
    ModelDiagnostics.analyze_model(model, tokenizer=tokenizer)

    # Log model parameter info
    logger.info(f"Trainable params: {model.print_trainable_parameters()}")
    
    # Load dataset
    train_dataset = JsonlDataset(
        dataset_path=data_args.dataset_path,
        tokenizer=tokenizer,
        template_yaml=data_args.template_path,
        max_seq_length=data_args.max_seq_length,
    )
    
    # Set debug flag if needed
    if model_args.enable_debug:
        train_dataset.enable_debug = True
        logger.info("Enabled debug mode for dataset")

    # Create custom data collator with padding and truncation
    logger.info("Setting up DataCollatorForCausalLM with padding")
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
        label_pad_token_id=-100
    )
    
    # Initialize Trainer
    callbacks = []
    if model_args.enable_debug:
        debug_callback = DebugCallback()
        callbacks.append(debug_callback)
        
        # Also add a callback to track batch-level details
        class BatchDebugCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                logger.info(f"Beginning step {state.global_step}")
                
            def on_step_end(self, args, state, control, **kwargs):
                if state.log_history and len(state.log_history) > 0:
                    for entry in reversed(state.log_history):
                        if 'loss' in entry:
                            logger.info(f"Step {state.global_step} completed with loss: {entry['loss']:.6f}")
                            break
                    
            def on_substep_end(self, args, state, control, **kwargs):
                logger.info(f"Substep completed in step {state.global_step}")
        
        callbacks.append(BatchDebugCallback())
        
        # Add a learning rate monitor callback
        class LRCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
                if optimizer:
                    for param_group in optimizer.param_groups:
                        logger.info(f"Current learning rate: {param_group['lr']:.8f}")
                        return
        
        callbacks.append(LRCallback())
    
    # Always add the LoRA activity tracking callback, even without debug mode
    # This is critical to diagnose why gradients aren't flowing
    lora_activity_callback = LoRAActivityCallback()
    callbacks.append(lora_activity_callback)
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Add DeepSpeed debugging code after setup but before training starts
    if model_args.enable_debug and training_args.deepspeed:
        logger.info("Adding special DeepSpeed hooks for debug mode")

        # Register a callback to add hooks once DeepSpeed is initialized
        class DeepSpeedDebuggerCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, **kwargs):
                logger.info("Training beginning, setting up DeepSpeed hooks...")
                
                # Run diagnostics on model after DeepSpeed wrapping
                logger.info("Running model diagnostics after DeepSpeed initialization...")
                ModelDiagnostics.analyze_model(model, tokenizer=trainer.tokenizer)
                
                # Check optimizer configuration
                if hasattr(trainer, 'optimizer'):
                    analyze_optimizer_and_scheduler(trainer)
                
                # Immediately verify that model still has LoRA parameters trainable
                def verify_model_trainable_params():
                    """Verify that only LoRA parameters are trainable after DeepSpeed init."""
                    trainable_params = []
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            trainable_params.append(name)
                    
                    lora_params = [p for p in trainable_params if any(lora_name in p for lora_name in ['lora_', 'adapter'])]
                    non_lora_params = [p for p in trainable_params if not any(lora_name in p for lora_name in ['lora_', 'adapter'])]
                    
                    logger.info(f"After DeepSpeed init: Found {len(lora_params)} LoRA trainable parameters")
                    if non_lora_params:
                        logger.error(f"❌ ERROR: Found {len(non_lora_params)} non-LoRA trainable parameters after DeepSpeed init!")
                        logger.error(f"First few: {non_lora_params[:5]}")
                        
                        # Re-freeze non-LoRA parameters
                        logger.info("Attempting to re-freeze non-LoRA parameters...")
                        for name, param in model.named_parameters():
                            if not any(lora_name in name for lora_name in ['lora_', 'adapter']):
                                param.requires_grad = False
                        
                        # Verify re-freezing was successful
                        still_trainable = [name for name, param in model.named_parameters() 
                                         if param.requires_grad and not any(lora_name in name for lora_name in ['lora_', 'adapter'])]
                        if still_trainable:
                            logger.error(f"❌ ERROR: Failed to freeze {len(still_trainable)} non-LoRA parameters")
                        else:
                            logger.info("✅ Successfully re-froze all non-LoRA parameters")
                
                # Verify model trainable parameters right away
                verify_model_trainable_params()
                
                # Function to add DeepSpeed hooks once the engine is available
                def setup_deepspeed_hooks():
                    if not hasattr(trainer, 'accelerator') or not hasattr(trainer.accelerator, 'deepspeed_engine'):
                        logger.warning("DeepSpeed engine not initialized yet, will try again later")
                        return False
                    
                    # Log DeepSpeed configuration
                    engine = trainer.accelerator.deepspeed_engine
                    config = engine.config
                    logger.info("DeepSpeed configuration:")
                    logger.info(f"- Zero stage: {config.get('zero_optimization', {}).get('stage', 'None')}")
                    logger.info(f"- Offload params: {config.get('zero_optimization', {}).get('offload_param', False)}")
                    logger.info(f"- FP16 enabled: {config.get('fp16', {}).get('enabled', False)}")
                    logger.info(f"- Initial loss scale: {config.get('fp16', {}).get('initial_scale_power', 'Not found')}")
                    logger.info(f"- Loss scale window: {config.get('fp16', {}).get('loss_scale_window', 'Not found')}")
                    
                    # Force check trainable parameters again after DeepSpeed initialization
                    verify_model_trainable_params()
                    
                    # Modify loss scale if needed
                    if config.get('fp16', {}).get('enabled', False):
                        # Check if loss scale is too high for Gemma
                        if hasattr(engine, 'optimizer') and hasattr(engine.optimizer, 'loss_scaler'):
                            loss_scaler = engine.optimizer.loss_scaler
                            if hasattr(loss_scaler, 'loss_scale'):
                                current_scale = loss_scaler.loss_scale
                                logger.info(f"Current loss scale: {current_scale}")
                                
                                # If loss scale is too high, reduce it (common issue with Gemma)
                                if current_scale > 2**16:
                                    new_scale = 2**16
                                    logger.warning(f"⚠️ Reducing loss scale from {current_scale} to {new_scale}")
                                    loss_scaler.loss_scale = new_scale
                                    logger.info(f"New loss scale: {loss_scaler.loss_scale}")
                    
                    # Monkey-patch DeepSpeed step function to print gradient information
                    if hasattr(engine, 'step'):
                        original_step = engine.step
                        
                        def debug_step():
                            logger.info("DeepSpeed step called - checking gradients before step")
                            # Log PEFT module gradients before step
                            grad_stats = {"has_grad": 0, "no_grad": 0, "zero_grad": 0, "non_zero_grad": 0}
                            for name, param in model.named_parameters():
                                if param.requires_grad and ('lora_' in name or 'adapter' in name):
                                    if param.grad is not None:
                                        grad_stats["has_grad"] += 1
                                        grad_norm = param.grad.data.norm().item()
                                        if grad_norm > 0:
                                            grad_stats["non_zero_grad"] += 1
                                            logger.info(f"✅ Non-zero gradient in {name}: {grad_norm:.6f}")
                                        else:
                                            grad_stats["zero_grad"] += 1
                                            logger.info(f"❌ Zero gradient in {name}")
                                    else:
                                        grad_stats["no_grad"] += 1
                                        logger.warning(f"No gradient computed for {name}")
                            
                            logger.info(f"Gradient stats: {grad_stats}")
                            
                            # Check loss scale
                            if hasattr(engine.optimizer, 'loss_scaler'):
                                loss_scaler = engine.optimizer.loss_scaler
                                if hasattr(loss_scaler, 'loss_scale'):
                                    logger.info(f"Current loss scale: {loss_scaler.loss_scale}")
                                if hasattr(loss_scaler, 'has_overflow_serial'):
                                    logger.info(f"Overflow detected: {loss_scaler.has_overflow_serial}")
                                    
                                    # If we detect overflow, reduce the loss scale
                                    if loss_scaler.has_overflow_serial and hasattr(loss_scaler, 'loss_scale'):
                                        current_scale = loss_scaler.loss_scale
                                        if current_scale > 1:
                                            new_scale = current_scale / 2
                                            logger.warning(f"⚠️ Overflow detected! Reducing loss scale to {new_scale}")
                                            loss_scaler.loss_scale = new_scale
                            
                            # Call original step
                            result = original_step()
                            logger.info("DeepSpeed step completed")
                            return result
                        
                        # Replace the step function
                        engine.step = debug_step
                        logger.info("Replaced DeepSpeed step function with debug version")
                    
                    # Hook into backward pass
                    if hasattr(engine, 'backward'):
                        original_backward = engine.backward
                        
                        def debug_backward(loss, **kwargs):
                            if hasattr(loss, 'item'):
                                logger.info(f"DeepSpeed backward called with loss: {loss.item()}")
                            result = original_backward(loss, **kwargs)
                            logger.info("DeepSpeed backward completed")
                            
                            # Check for gradients after backward
                            params_with_grad = 0
                            for name, param in model.named_parameters():
                                if param.requires_grad and param.grad is not None:
                                    params_with_grad += 1
                            
                            logger.info(f"After backward: {params_with_grad} parameters have gradients")
                            return result
                        
                        engine.backward = debug_backward
                        logger.info("Replaced DeepSpeed backward function with debug version")
                    
                    return True
                    
                # Try to set up hooks now
                if not setup_deepspeed_hooks():
                    # If not successful, try again after a step begins
                    logger.info("Will try to set up DeepSpeed hooks again after first step")
            
            def on_step_begin(self, args, state, control, model=None, **kwargs):
                if state.global_step == 0:
                    logger.info("First step beginning, setting up DeepSpeed hooks if not already done")
                    # Function from on_train_begin
                    if hasattr(self, 'setup_deepspeed_hooks'):
                        self.setup_deepspeed_hooks()
        
        # Add the callback
        trainer.add_callback(DeepSpeedDebuggerCallback())
    
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

# Add a special callback to verify LoRA layer activity during forward passes
class LoRAActivityCallback(TrainerCallback):
    """Monitors LoRA layer activations during forward passes to verify they're being used."""
    
    def __init__(self):
        self.hooks = []
        self.activation_counts = {}
        self.total_calls = 0
        self.hook_registered = False
        self.first_batch = True
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Register hooks on LoRA modules to track their usage."""
        if model is None:
            return
        
        logger.info("Setting up LoRA activity tracking hooks...")
        
        self._register_hooks(model)
    
    def _register_hooks(self, model):
        """Register forward hooks on all LoRA modules."""
        if self.hook_registered:
            return
        
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Find all LoRA modules
        lora_modules = {}
        for name, module in model.named_modules():
            if 'lora_' in name:
                # Add to our tracking dict
                if name not in self.activation_counts:
                    self.activation_counts[name] = 0
                lora_modules[name] = module
        
        if not lora_modules:
            logger.warning("No LoRA modules found to monitor!")
            return
        
        logger.info(f"Found {len(lora_modules)} LoRA modules to monitor")
        
        # Define a hook function to count activations
        def hook_fn(module_name):
            def hook(module, input, output):
                self.activation_counts[module_name] += 1
                return output
            return hook
        
        # Register hooks for each LoRA module
        for name, module in lora_modules.items():
            h = module.register_forward_hook(hook_fn(name))
            self.hooks.append(h)
        
        self.hook_registered = True
        logger.info("LoRA activation tracking hooks registered successfully")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Check activation counts periodically."""
        self.total_calls += 1
        
        # Report on first batch and then every 10 steps
        if self.first_batch or self.total_calls % 10 == 0:
            self._report_activations()
            
            if self.first_batch:
                self.first_batch = False
    
    def _report_activations(self):
        """Report activation statistics for LoRA modules."""
        active_modules = sum(1 for count in self.activation_counts.values() if count > 0)
        total_modules = len(self.activation_counts)
        
        logger.info(f"LoRA module activation report (step {self.total_calls}):")
        logger.info(f"- Active modules: {active_modules}/{total_modules}")
        
        if active_modules == 0:
            logger.error("❌ NO LORA MODULES ACTIVE - model is not using LoRA layers!")
            logger.error("This confirms LoRA is not integrated into computation graph")
            
            # Provide some examples of inactive modules
            inactive = [name for name, count in self.activation_counts.items() if count == 0]
            if inactive:
                logger.error(f"Examples of inactive modules: {inactive[:5]}")
        else:
            logger.info(f"✅ {active_modules} LoRA modules are being activated during forward pass")
            
            # Show some of the most active modules
            active = sorted(
                [(name, count) for name, count in self.activation_counts.items() if count > 0],
                key=lambda x: x[1],
                reverse=True
            )
            
            for name, count in active[:5]:
                logger.info(f"  - {name}: {count} activations")
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Clean up hooks at end of training."""
        logger.info("Cleaning up LoRA activation monitoring hooks")
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Final activation report
        logger.info("Final LoRA activation report:")
        active_modules = sum(1 for count in self.activation_counts.values() if count > 0)
        total_modules = len(self.activation_counts)
        
        logger.info(f"- Active modules: {active_modules}/{total_modules}")
        if active_modules == 0:
            logger.error("❌ TRAINING COMPLETED WITH NO LORA ACTIVATIONS!")
        else:
            logger.info(f"✅ Training successfully used {active_modules} LoRA modules")

if __name__ == "__main__":
    main() 