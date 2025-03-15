#!/bin/bash

set -eu

echo "Starting Gemma-3 SFT training on a single GPU (no distributed)..."

# Create directories (with configurable paths)
OUTPUT_DIR=${OUTPUT_DIR:-"./gemma3_sft_output"}
TEMPLATE_PATH=${TEMPLATE_PATH:-"templates/borg_chat_exp.yaml"}
DATA_PATH=${DATA_PATH:-"./data/conversations.jsonl"}

# Make sure directories exist
mkdir -p $OUTPUT_DIR

# Create a simple data file for testing if needed
if [ ! -f "$DATA_PATH" ]; then
    echo "Creating a simple test dataset at $DATA_PATH"
    mkdir -p $(dirname "$DATA_PATH")
    cat > "$DATA_PATH" << EOL
{"instruction": "Tell me about quantum computing", "input": "", "output": "Quantum computing is a type of computing that uses quantum mechanics to process information. Traditional computers use bits, which can be either 0 or 1, while quantum computers use quantum bits or qubits that can exist in multiple states simultaneously due to superposition. This allows quantum computers to perform certain calculations much faster than classical computers."}
{"instruction": "Explain how to make pancakes", "input": "", "output": "To make pancakes, you'll need flour, eggs, milk, baking powder, sugar, and salt. Mix the dry ingredients first, then add the wet ingredients and stir until just combined (small lumps are okay). Heat a lightly oiled griddle or pan over medium-high heat. Pour about 1/4 cup of batter onto the griddle for each pancake. Cook until bubbles form and edges look dry, then flip and cook for another minute."}
EOL
fi

# Create a simple template file for testing if needed
if [ ! -f "$TEMPLATE_PATH" ]; then
    echo "Creating a simple test template at $TEMPLATE_PATH"
    mkdir -p $(dirname "$TEMPLATE_PATH")
    cat > "$TEMPLATE_PATH" << EOL
sequence:
  - no_loss: "<bos>"
  - no_loss: "Instruction: {instruction}\n"
  - no_loss: "Input: {input}\n"
  - no_loss: "Output: "
  - loss: "{output}"
  - loss: "<eos>"
EOL
fi

echo "Using:"
echo "- Output directory: $OUTPUT_DIR"
echo "- Template: $TEMPLATE_PATH"
echo "- Data: $DATA_PATH"

# Run training directly (no distributed runtime)
# We explicitly avoid using DeepSpeed or torch.distributed.run
python gemma_sft_train.py \
    --model_name_or_path "google/gemma-3-27b-pt" \
    --dataset_path $DATA_PATH \
    --template_path $TEMPLATE_PATH \
    --max_seq_length 512 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --save_steps 50 \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --overwrite_output_dir \
    --gradient_checkpointing True \
    --save_total_limit 1 \
    --max_grad_norm 1.0 \
    --no_cuda False \
    --local_rank -1

echo "Training complete!" 