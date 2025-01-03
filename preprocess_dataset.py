import os
import logging
import numpy as np
from tqdm import tqdm
import yaml
from transformers import AutoTokenizer
import jax
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

def process_example(example, tokenizer, template):
    """Process a single example using the template."""
    # Parse template
    template_data = yaml.safe_load(template)
    
    # Initialize buffers
    token_buffer = []
    loss_mask_buffer = []
    
    # Process each segment
    for segment in template_data['sequence']:
        for loss_key, content in segment.items():
            # Set loss mask based on key
            mask = 0.0 if loss_key == 'no_loss' else 1.0
            
            # Format content with example values
            text = content.format(**example)
            
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)
            loss_mask_buffer.extend([mask] * len(tokens))
    
    return {
        'tokens': np.array(token_buffer, dtype=np.int32),
        'loss_masks': np.array(loss_mask_buffer, dtype=np.float32)
    }

def preprocess_dataset(
    input_path='tatsu-lab/alpaca',
    output_dir='/mnt/disk2/preprocessed_datasets',
    template_path='templates/llama_chat.yaml',
    tokenizer_name='meta-llama/Llama-3.2-1B',
    batch_size=1000,
    max_length=2048
):
    """Preprocess an entire dataset and save in Parquet format."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load template
    with open(template_path) as f:
        template = f.read()
    
    # Load dataset
    dataset = load_dataset(input_path, streaming=False)['train']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in batches
    processed_examples = []
    total_examples = len(dataset)
    
    for i in tqdm(range(0, total_examples, batch_size)):
        batch = dataset[i:i + batch_size]
        
        # Process each example
        for example in batch:
            processed = process_example(example, tokenizer, template)
            
            # Skip if too long
            if len(processed['tokens']) > max_length:
                continue
                
            # Pad to max_length
            tokens = np.pad(
                processed['tokens'],
                (0, max_length - len(processed['tokens'])),
                mode='constant',
                constant_values=tokenizer.pad_token_id
            )
            loss_masks = np.pad(
                processed['loss_masks'],
                (0, max_length - len(processed['loss_masks'])),
                mode='constant',
                constant_values=0
            )
            
            processed_examples.append({
                'tokens': tokens,
                'loss_masks': loss_masks
            })
        
        # Write batch to parquet when we have enough examples
        if len(processed_examples) >= batch_size:
            # Convert to Arrow table
            table = pa.Table.from_pylist(processed_examples)
            
            # Write to parquet
            output_path = os.path.join(
                output_dir,
                f'processed_{i//batch_size:05d}.parquet'
            )
            pq.write_table(table, output_path)
            
            # Clear buffer
            processed_examples = []
    
    # Write any remaining examples
    if processed_examples:
        table = pa.Table.from_pylist(processed_examples)
        output_path = os.path.join(
            output_dir,
            f'processed_{(i//batch_size)+1:05d}.parquet'
        )
        pq.write_table(table, output_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    preprocess_dataset()
