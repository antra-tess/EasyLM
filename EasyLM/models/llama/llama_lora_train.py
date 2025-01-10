import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/mnt/disk2/jax_cache"

import pprint

from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu
import logging
import os

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from typing import Any
import flax.struct
import optax

jax.config.update("jax_compilation_cache_dir", "/mnt/disk2/jax_cache")
#jax.config.update("jax_explain_cache_misses", True)
#jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

class LoRATrainState(flax.struct.PyTreeNode):
    """Simple train state for LoRA training."""
    step: int
    params: Any
    opt_state: Any
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    
    def apply_gradients(self, *, grads, **kwargs):
        """Apply gradients to state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs
        )

    @classmethod
    def create(cls, *, params, tx, **kwargs):
        """Creates a new instance with parameters and their optimizer states.
        
        For LoRA training, we only create optimizer states for LoRA parameters.
        Base model parameters are kept but won't have associated optimizer state.
        """
        # Initialize optimizer state only for LoRA parameters
        flat_params = flatten_dict(params)
        lora_params = {}
        for k, v in flat_params.items():
            path_str = '/'.join(str(x) for x in k)
            if 'lora_' in path_str:
                lora_params[k] = v
        
        # Only create optimizer state for LoRA parameters
        lora_params = unflatten_dict(lora_params)
        opt_state = tx.init(lora_params)

        if jax.process_index() == 0:
            logging.info(f"Created optimizer state for LoRA parameters")
            
        return cls(
            step=0,
            params=params,  # Keep full params
            opt_state=opt_state,  # But only optimizer state for LoRA
            tx=tx,
            **kwargs
        )

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import AutoTokenizer
import optax

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, make_shard_and_gather_fns,
    with_sharding_constraint, named_tree_map,
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfigurator, FlaxLLaMAForCausalLMModule
)

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    dtype='fp32',
    param_dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer='openlm-research/open_llama_3b_v2',
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfigurator.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    checkpointing=mlxu.config_dict(
        save_min_step=100,
        save_loss_threshold=3,
        keep_recent=25,  # Number of recent checkpoints to keep
    ),
)


def set_in_dict(d, path, value):
    """Sets a value in a nested dictionary using a path tuple."""
    try:
        keys = path if isinstance(path, tuple) else (path,)
        d_ptr = d
        for key in keys[:-1]:
            d_ptr = d_ptr[key]
        d_ptr[keys[-1]] = value
        return d
    except KeyError as e:
        raise KeyError(f"Key not found: {e}, path: {path}, dict: {d}")

def logginginfo(msg, *args):
    if jax.process_index() == 0:
        logging.info(msg, *args)

def main(argv):
    # Set JAX logging to DEBUG to see compilation logs
    #logging.getLogger("jax").setLevel(logging.DEBUG)
    
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    
    # Log process/worker information
    hostname = os.uname().nodename
    process_index = jax.process_index()
    process_count = jax.process_count()
    logging.info(f"Starting up on host {hostname} - process index: {process_index}/{process_count}")
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)
    
    # # Add base model name as a tag
    # if not hasattr(FLAGS.logger, 'tags') or FLAGS.logger.tags is None:
    #     FLAGS.logger.tags = []
    # FLAGS.logger.tags.append(f"base_model_{llama_config.base_model}")

    logginginfo("Starting model initialization...")
    model = FlaxLLaMAForCausalLMModule(
        llama_config,
        dtype=get_float_dtype_by_name(FLAGS.dtype),
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )
    logginginfo(f"Model initialization complete: LLaMA {llama_config.base_model}")


    logginginfo("Setting up optimizer...")
    # Use trainable_mask for both weight decay and controlling optimizer state allocation
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer
    )
    logginginfo(f"Optimizer setup complete: {str(optimizer_info)}, {str(optimizer)}, {str(type(optimizer))}")

    def create_trainstate_from_params(params):
        train_state = LoRATrainState.create(params=params, tx=optimizer)
        return train_state

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        logginginfo("Getting model parameter structure...")
        full_param_shapes = jax.eval_shape(
            lambda: model.init(
                input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
                position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
                rngs=rng_generator(LLaMAConfigurator.rng_keys()),
            )
        )
        
        logginginfo("Initializing LoRA parameters from model structure...")
        # Create empty LoRA parameter structure matching model structure
        flat_shapes = flatten_dict(full_param_shapes['params'])
        lora_params = {}
        for path, shape in flat_shapes.items():
            path_str = '/'.join(str(x) for x in path)
            # Check if this is an attention weight we want to add LoRA to
            if 'attention' in path_str and any(f'/{w}/' in path_str for w in ['wq', 'wk', 'wv', 'wo']) and path[-1] == 'kernel':
                if llama_config.lora_attn:
                    # Get the module path without the final 'kernel'
                    base_path = path[:-1]
                    # Add LoRA A and B parameters
                    # Initialize LoRA parameters with small normal values like in test
                    # Use a single RNG split for both A and B to avoid tracer issues
                    rng_split = jax.random.split(rng, 2)
                    lora_A_path = base_path + ('lora_A',)
                    lora_B_path = base_path + ('lora_B',)
                    lora_params[lora_A_path] = jax.random.normal(
                        rng_split[0], (shape.shape[0], llama_config.lora_rank)
                    ) * 0.02
                    lora_params[lora_B_path] = jax.random.normal(
                        rng_split[1], (llama_config.lora_rank, shape.shape[1])
                    ) * 0.02
            # Check if this is a MLP weight we want to add LoRA to
            elif 'feed_forward' in path_str and any(f'/{w}/' in path_str for w in ['w1', 'w2', 'w3']) and path[-1] == 'kernel':
                if llama_config.lora_mlp:
                    base_path = path[:-1]
                    lora_A_path = base_path + ('lora_A',)
                    lora_B_path = base_path + ('lora_B',)
                    lora_params[lora_A_path] = jnp.zeros((shape.shape[0], llama_config.lora_rank))
                    lora_params[lora_B_path] = jnp.zeros((llama_config.lora_rank, shape.shape[1]))

        # Add debug logging
        if jax.process_index() == 0:
            logginginfo("LoRA parameter paths:")
            for path in lora_params.keys():
                logginginfo(f"  {'/'.join(str(x) for x in path)}")

        # Don't wrap in params dict since model.apply will do that
        lora_params = unflatten_dict(lora_params)
        
        logginginfo("Creating train state from LoRA parameters...")
        train_state = LoRATrainState.create(params=lora_params, tx=optimizer)
        logginginfo("Train state creation complete")
        return train_state

    def combine_params(base_params, lora_params):
        """Combine base_params and lora_params back into a single param tree."""
        # Base case: if either input is not a dict, return base params
        if 'params' in base_params:
            base_params = base_params['params']

        combined = {}
        # First add all base params
        for k, v in base_params.items():
            combined[k] = v
        # Then overlay LoRA params
        for k, v in lora_params.items():
            if k in combined:
                combined[k] = combine_params(combined[k], v)
            else:
                combined[k] = v
        return combined

    # Quick test of combine_params
    if jax.process_index() == 0:
        logging.info("Testing combine_params...")
    test_base = {'params': {'transformer': {'h': {'0': {'kernel': jnp.ones((2,2))}}}, 'lm_head': {'kernel': jnp.ones((2,2))}}}
    test_lora = {'params': {'transformer': {'h': {'0': {'lora_A': jnp.zeros((2,2))}}}, 'lm_head': {'kernel': jnp.zeros((2,2))}}}
    try:
        test_combined = combine_params(test_base, test_lora)
        if jax.process_index() == 0:
            logging.info("combine_params test passed")
            logging.info(f"Combined structure: {jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, test_combined)}")
    except Exception as e:
        logging.error(f"combine_params test failed: {e}")
        raise

    def train_step(train_state, base_params, rng, batch):
        """Training step with separate base and LoRA parameters.
    
        Args:
            train_state: Contains LoRA parameters and optimizer state
            base_params: Base model parameters (frozen)
            rng: Random number generator key
            batch: Training batch
        """
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
    
        def loss_and_accuracy(lora_params):
            # Combine with base params for forward pass
            params = {'params': combine_params(base_params, lora_params)}
            logits = model.apply(
                params, batch['input_tokens'], deterministic=False,
                rngs=rng_generator(LLaMAConfigurator.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(
                logits, batch['target_tokens'], batch['loss_masks']
            )

        # Compute gradients only for LoRA params
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        logginginfo("Gradient computation complete")
        # Count total and non-zero gradients
        flat_grads = flatten_dict(grads)
        num_nonzero = sum(jnp.any(jnp.abs(v) > 0) for v in flat_grads.values())

        # Update LoRA params only
        updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
        logginginfo("Optimizer state update complete")
        new_params = optax.apply_updates(train_state.params, updates)
        logginginfo("LoRA parameter updated from optimizer state")
        train_state = train_state.replace(
            step=train_state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
        logginginfo("Train state updated")


        # Compute gradient statistics using JAX ops
        grad_norms = jnp.array([jnp.linalg.norm(v) for v in flat_grads.values()])
        grad_max = jnp.max(grad_norms)
        grad_mean = jnp.mean(grad_norms)
        
        # Calculate separate gradient stats for MLP and attention
        mlp_grads = []
        attn_grads = []
        for key, value in flat_grads.items():
            path = '/'.join(str(x) for x in key)
            if 'lora_' in path:
                if 'feed_forward' in path:
                    mlp_grads.append(jnp.mean(jnp.abs(value)))
                elif 'attention' in path:
                    attn_grads.append(jnp.mean(jnp.abs(value)))

        mlp_grad_mean = jnp.mean(jnp.array(mlp_grads)) if mlp_grads else 0
        attn_grad_mean = jnp.mean(jnp.array(attn_grads)) if attn_grads else 0

        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
            num_grads=len(flat_grads),
            num_nonzero_grads=num_nonzero,
            grad_max=grad_max,
            grad_mean=grad_mean,
            mlp_grad_mean=mlp_grad_mean,
            attn_grad_mean=attn_grad_mean
        )
        rng = rng_generator()
        logginginfo("Train step complete")
        return train_state, rng, metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            train_state.params, batch['input_tokens'], deterministic=True,
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    logginginfo("Calculating partitioning...")
    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    logginginfo("Train state shape calculation complete")
    # Get partition specs for LoRA train state
    train_state_partition = match_partition_rules(
        LLaMAConfigurator.get_lora_partition_rules(), train_state_shapes
    )
    
    # Get partition specs for base parameters
    init_rng = next_rng()
    # Get full model shapes first
    full_param_shapes = jax.eval_shape(
        lambda: model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=JaxRNG(init_rng)(LLaMAConfigurator.rng_keys()),
        )
    )
    
    # Extract just base parameter shapes by removing LoRA parameters
    flat_shapes = flatten_dict(full_param_shapes)
    base_shapes = {}
    if jax.process_index() == 0:
        logginginfo("Full parameter shapes:")
        for k, v in flat_shapes.items():
            path_str = '/'.join(str(x) for x in k)
            logginginfo(f"  {path_str}: {v.shape if hasattr(v, 'shape') else v}")
    
    for k, v in flat_shapes.items():
        path_str = '/'.join(str(x) for x in k)
        if 'lora_' not in path_str:
            base_shapes[k] = v
            
    if jax.process_index() == 0:
        logginginfo("Base parameter shapes after filtering:")
        for k, v in base_shapes.items():
            path_str = '/'.join(str(x) for x in k)
            logginginfo(f"  {path_str}: {v.shape if hasattr(v, 'shape') else v}")
            
        # Specifically check for embedding
        logginginfo("\nChecking for embedding parameters:")
        for k, v in base_shapes.items():
            if 'wte' in str(k):
                logginginfo(f"Found embedding parameter: {k}")
            
    base_param_shapes = unflatten_dict(base_shapes)
    
    # Now get partition rules for just base parameters
    base_param_partition = match_partition_rules(
        LLaMAConfigurator.get_base_param_rules(), base_param_shapes, debug_print=True,
    )
    
    # Keep original nested structure for base parameter partitioning
    base_param_partition = match_partition_rules(
        LLaMAConfigurator.get_base_param_rules(), base_param_shapes, debug_print=True,
    )
    logginginfo("Train state partitioning complete")

    # # Log partition specs and actual shapes
    # if jax.process_index() == 0:
    #     logginginfo("Examining train state partitioning:")
    #     # Flatten each field of TrainState separately
    #     for field in ["params", "opt_state", "step"]:
    #         logginginfo(f"\nExamining {field}:")
    #         field_partition = getattr(train_state_partition, field)
    #         field_shapes = getattr(train_state_shapes, field)
    #         if isinstance(field_partition, (dict, FrozenDict)):
    #             flat_partition = flatten_dict(field_partition)
    #             flat_shapes = flatten_dict(field_shapes)
    #             for name, spec in flat_partition.items():
    #                 shape = flat_shapes[name].shape if hasattr(flat_shapes[name], 'shape') else None
    #                 logginginfo(f"Parameter {name}:")
    #                 logginginfo(f"  Shape: {shape}")
    #                 logginginfo(f"  Partition spec: {spec}")
    #         else:
    #             logginginfo(f"  Shape: {getattr(field_shapes, 'shape', None)}")
    #             logginginfo(f"  Partition spec: {field_partition}")

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    logginginfo("Setting up partitioned functions...")
    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )
    logginginfo("Partitioned initialization function created")

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    # Print full tree structures before pjit
    if jax.process_index() == 0:
        logginginfo("Train state partition structure:")
        logginginfo(str(jax.tree_util.tree_structure(train_state_partition)))
        logginginfo("Base param partition structure:")
        logginginfo(str(jax.tree_util.tree_structure(base_param_partition)))

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, base_param_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0,),  # Only donate train_state memory
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )
    logginginfo("Partitioned functions created")


    def prune_none(tree):
        """Recursively remove any `None` leaves to produce a smaller dict."""
        if isinstance(tree, dict):
            new_dict = {}
            for k, v in tree.items():
                pruned_v = prune_none(v)
                if pruned_v is not None:
                    new_dict[k] = pruned_v
            return new_dict if new_dict else None
        elif isinstance(tree, (tuple, list)):
            new_list = []
            for v in tree:
                pruned_v = prune_none(v)
                if pruned_v is not None:
                    new_list.append(pruned_v)
            return type(tree)(new_list) if new_list else None
        else:
            return tree  # None or actual param


    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        # Skip early checkpoints unless it's a milestone
        if not milestone and step < FLAGS.checkpointing.save_min_step:
            return
            
        # Get current loss - skip if above threshold
        metrics = jax.device_get(train_state.metrics) if hasattr(train_state, 'metrics') else None
        if not milestone and metrics and metrics.get('loss', 0) > FLAGS.checkpointing.save_loss_threshold:
            return
            
        hostname = os.uname().nodename
        process_index = jax.process_index()
        logginginfo(f"Checkpoint save called on host {hostname} (process {process_index}) at step {step}...")

        # Extract only LoRA parameters for saving
        flat_params = flatten_dict(train_state.params)
        lora_params = {}
        for k, v in flat_params.items():
            if 'lora_' in str(k):
                lora_params[k] = v
        lora_params = unflatten_dict(lora_params)

        # Create partial state with only LoRA parameters
        partial_state = train_state.replace(params={'params': lora_params})

        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        
        checkpoint_dir = os.path.join(logger.output_dir, f"checkpoint_{step}")
        if milestone:
            checkpoint_dir = os.path.join(logger.output_dir, f"milestone_{step}")
        logginginfo(f"Saving checkpoint to: {checkpoint_dir}")

        # Cleanup old checkpoints if needed
        if not milestone and FLAGS.checkpointing.keep_recent > 0:
            # List all regular checkpoints
            checkpoints = []
            for fname in os.listdir(logger.output_dir):
                if fname.startswith('checkpoint_'):
                    try:
                        step_num = int(fname.split('_')[1])
                        checkpoints.append((step_num, fname))
                    except (IndexError, ValueError):
                        continue
            
            # Sort by step number and remove old ones
            checkpoints.sort(reverse=True)  # Newest first
            for _, fname in checkpoints[FLAGS.checkpointing.keep_recent:]:
                old_checkpoint = os.path.join(logger.output_dir, fname)
                logginginfo(f"Removing old checkpoint: {old_checkpoint}")
                os.system(f"rm -rf {old_checkpoint}")
        # For LoRA training, we need to adjust gather_fns structure
        lora_gather_fns = gather_fns  # Keep original structure since we already wrapped params
        checkpointer.save_all(
            train_state=partial_state,
            gather_fns=lora_gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
            params_wrapped=False,
        )
        logginginfo("Checkpoint save complete")

    logginginfo("Setting up JAX mesh...")
    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    logginginfo("JAX mesh initialized")
    def log_memory_usage(prefix=""):
        if jax.process_index() == 0:
            logging.info(f"\n{prefix} Memory Report:")
            
            try:
                live_arrays = jax.live_arrays()
                total_live = sum(x.nbytes for x in live_arrays) / 1e9
                logging.info(f"Total live array bytes: {total_live:.2f} GB")
                
                # Group arrays by device and type
                arrays_by_device = {}
                arrays_by_type = {}
                total_arrays = len(live_arrays)
                
                for arr in live_arrays:
                    # Get device info safely
                    try:
                        if hasattr(arr, 'device_buffer'):
                            device = str(arr.device_buffer.device())
                        elif hasattr(arr, 'devices'):
                            device = str(list(arr.devices())[0])  # Take first device if sharded
                        elif hasattr(arr, 'device'):
                            device = str(arr.device())
                        elif hasattr(arr, '_device'):
                            device = str(arr._device)
                        elif isinstance(arr, jax.Array):
                            # device_set is a set, convert to list to index
                            device = str(list(arr.sharding.device_set)[0])
                        else:
                            device = 'unknown'
                    except Exception as e:
                        if jax.process_index() == 0:
                            logging.info(f"Error getting device info: {str(e)}")
                        device = 'unknown'
                        
                    # Add to device grouping
                    if device not in arrays_by_device:
                        arrays_by_device[device] = []
                    arrays_by_device[device].append(arr)
                    
                    # Add to shape pattern grouping
                    if hasattr(arr, 'shape'):
                        shape_pattern = 'x'.join(str(x) for x in arr.shape)
                    else:
                        shape_pattern = 'unknown'
                    if shape_pattern not in arrays_by_type:
                        arrays_by_type[shape_pattern] = []
                    arrays_by_type[shape_pattern].append(arr)

                # Print summary statistics
                logging.info(f"\nTotal number of arrays: {total_arrays}")
                
                # Print device statistics
                logging.info("\nMemory by device:")
                for device, arrays in sorted(arrays_by_device.items()):
                    device_total = sum(x.nbytes for x in arrays) / 1e9
                    array_count = len(arrays)
                    logging.info(f"\nDevice {device}:")
                    logging.info(f"  Total memory: {device_total:.2f} GB")
                    logging.info(f"  Array count: {array_count}")
                    logging.info("  Top 5 largest arrays:")
                    for arr in sorted(arrays, key=lambda x: x.nbytes, reverse=True)[:5]:
                        logging.info(f"    shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes / 1e9:.2f} GB")
                
                # Print shape pattern statistics
                logging.info("\nArrays grouped by shape pattern:")
                for shape_pattern, arrays in sorted(arrays_by_type.items(), 
                                                 key=lambda x: sum(a.nbytes for a in x[1]), 
                                                 reverse=True)[:10]:
                    total_size = sum(x.nbytes for x in arrays) / 1e9
                    array_count = len(arrays)
                    if array_count > 0:  # Only show non-empty groups
                        logging.info(f"\nShape pattern: {shape_pattern}")
                        logging.info(f"  Count: {array_count}")
                        logging.info(f"  Total size: {total_size:.2f} GB")
                        logging.info(f"  Example dtype: {arrays[0].dtype}")

            except Exception as e:
                logging.info(f"Unable to get live array info: {str(e)}")
                import traceback
                logging.info(f"Traceback: {traceback.format_exc()}")

    with mesh:
        train_state, restored_params = None, None
        log_memory_usage("Before checkpoint load")
        
        if FLAGS.load_checkpoint != '':
            logginginfo("Loading checkpoint...")
            # Get base parameter sharding functions using base rules
            base_shard_fns = make_shard_and_gather_fns(
                match_partition_rules(
                    LLaMAConfigurator.get_base_param_rules(),
                    base_param_shapes
                ),
                base_param_shapes
            )[0]

            # Load base model parameters first
            base_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, base_param_shapes, base_shard_fns
            )
            logginginfo("Loaded base model parameters")
            
            
            log_memory_usage("After base parameter load")

        if train_state is None and restored_params is None:
            # Initialize from scratch
            logginginfo("Initializing from scratch")
            log_memory_usage("Before initialization")
            train_state = sharded_init_fn(next_rng())
            log_memory_usage("After initialization")
            logginginfo("Initialization complete")
        elif train_state is None and restored_params is not None:
            # Initialize only LoRA params, use restored params as base
            logginginfo(f"Initializing LoRA parameters with rank {llama_config.lora_rank}")
            log_memory_usage("Before LoRA init")
            
            # Validate LoRA config
            if llama_config.lora_rank <= 0:
                raise ValueError(f"LoRA rank must be positive, got {llama_config.lora_rank}")
            if llama_config.lora_alpha <= 0:
                raise ValueError(f"LoRA alpha must be positive, got {llama_config.lora_alpha}")
            if not (llama_config.lora_attn or llama_config.lora_mlp):
                raise ValueError("At least one of lora_attn or lora_mlp must be True")
            
            # Create empty LoRA parameter structure
            lora_params = {'params': {}}
            for layer_idx in range(llama_config.num_hidden_layers):
                layer_params = {}
                # Attention LoRA params
                if llama_config.lora_attn:
                    for name in ['wq', 'wk', 'wv', 'wo']:
                        layer_params[f'attention/{name}/lora_A'] = jnp.zeros((llama_config.hidden_size, llama_config.lora_rank))
                        layer_params[f'attention/{name}/lora_B'] = jnp.zeros((llama_config.lora_rank, llama_config.hidden_size))
                # MLP LoRA params if enabled
                if llama_config.lora_mlp:
                    for name in ['w1', 'w2', 'w3']:
                        layer_params[f'feed_forward/{name}/lora_A'] = jnp.zeros((llama_config.hidden_size, llama_config.lora_rank))
                        layer_params[f'feed_forward/{name}/lora_B'] = jnp.zeros((llama_config.lora_rank, llama_config.hidden_size))
                lora_params['params'][f'transformer/h/{layer_idx}'] = layer_params
                
            # Validate created parameter structure
            flat_params = flatten_dict(lora_params)
            expected_param_count = llama_config.num_hidden_layers * (
                (8 if llama_config.lora_attn else 0) +  # 4 attention matrices * 2 (A&B)
                (6 if llama_config.lora_mlp else 0)     # 3 MLP matrices * 2 (A&B)
            )
            actual_param_count = len([k for k in flat_params.keys() if 'lora_' in str(k)])
            if actual_param_count != expected_param_count:
                raise ValueError(f"Expected {expected_param_count} LoRA parameters but got {actual_param_count}")
            
            logginginfo(f"Created LoRA parameter structure with {actual_param_count} parameters")
            
            # Create train state with only LoRA params
            logginginfo("Creating train state from LoRA params")
            train_state = sharded_create_trainstate_from_params(lora_params)
            
            # Use restored params as base
            base_params = restored_params
            logginginfo("Using restored checkpoint as base parameters")
            log_memory_usage("After LoRA init")
            logginginfo("Train state creation complete")

        # Print sharded parameter info on worker 0 only
        if jax.process_index() == 0:
            def print_params_tree(tree, path=''):
                if isinstance(tree, dict):
                    for key, value in tree.items():
                        new_path = f"{path}/{key}" if path else key
                        print_params_tree(value, new_path)
                else:
                    shape_dtype = jax.eval_shape(lambda: tree)
                    logginginfo(f"Parameter: {path} with shape {shape_dtype.shape} and sharding {tree.sharding}")

            logginginfo("Model parameters after sharding:")
            print_params_tree(train_state.params)
        logginginfo("Getting start step")

        start_step = int(jax.device_get(train_state.step))


        if FLAGS.save_model_freq > 0:
            logginginfo("Before saving checkpoint")
            save_checkpoint(train_state)
            logginginfo("After saving checkpoint")


        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        log_memory_usage("Before training loop")
        logginginfo("Starting training loop...")
        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, base_params, sharded_rng, batch
            )
            logginginfo("Train step done")

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)
