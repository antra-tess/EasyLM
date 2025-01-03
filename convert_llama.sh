python -m EasyLM.models.llama.convert_hf_to_easylm \
     --hf_model=/mnt/disk2/llama-3.2-1b \
     --output_file=/mnt/disk2/llama-3.2-1b.easylm \
     --llama.base_model=llama32_1b \
     --streaming \
     --float_dtype=bf16
