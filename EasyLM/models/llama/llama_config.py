import mlxu
from EasyLM.models.llama.llama_model import LLaMAConfigurator
from EasyLM.serving import LMServer
from EasyLM.jax_utils import JaxDistributedConfig

# Define flags only once
FLAGS = None
FLAGS_DEF = None


def create_llama_flags(updates=None):
    """Create standard LLaMA flags with optional updates."""
    global FLAGS, FLAGS_DEF

    if FLAGS is None:
        # Only define flags the first time
        FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
            seed=42,
            mesh_dim='1,-1,1',
            param_dtype='bf16',
            dtype='bf16',
            input_length=1024,
            seq_length=2048,
            top_k=50,
            top_p=1.0,
            do_sample=True,
            num_beams=1,
            add_bos_token=True,
            load_checkpoint='',
            tokenizer='openlm-research/open_llama_3b_v2',
            llama=LLaMAConfigurator.get_default_config(),
            lm_server=LMServer.get_default_config(),
            jax_distributed=JaxDistributedConfig.get_default_config(),
            coordinator_url='http://localhost:5010'  # Add coordinator URL flag
        )

    if updates is not None:
        for key, value in updates.items():
            if hasattr(FLAGS, key):
                setattr(FLAGS, key, value)

    return FLAGS, FLAGS_DEF
