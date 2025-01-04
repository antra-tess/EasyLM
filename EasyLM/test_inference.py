import logging
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from EasyLM.models.llama.llama_serve import ModelServer, FLAGS
from EasyLM.models.llama.llama_config import create_llama_flags

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Initialize flags with same values as worker_serve.sh
    FLAGS, _ = create_llama_flags({
        'mesh_dim': '1,-1,1',
        'dtype': 'bf16',
        'llama': {'base_model': 'llama32_1b'},
        'tokenizer': 'meta-llama/Llama-3.2-1B',
        'load_checkpoint': 'base_params_unsharded::/mnt/disk2/llama-3.2-1b.easylm',
        'input_length': 1024,
        'seq_length': 2048,
        'do_sample': True,
        'top_k': 50,
        'top_p': 0.95,
    })
    
    # Initialize model server
    server = ModelServer(FLAGS.lm_server)
    
    # Run inference on fixed text
    test_text = ["Tell me a short story about a cat."]
    response = server.generate(test_text, temperature=1.0)
    
    print("\nInput:", test_text[0])
    print("\nOutput:", response[0])
