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
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import AutoTokenizer

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

def main(argv):
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

    logging.info("Starting model initialization...")
    model = FlaxLLaMAForCausalLMModule(
        llama_config,
        dtype=get_float_dtype_by_name(FLAGS.dtype),
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )
    logging.info(f"Model initialization complete: LLaMA {llama_config.base_model}")

    def trainable_mask(param_name: str, param_value=None) -> bool:
        """
        If LoRA is off (lora_rank=0), we return True for all parameters.
        If LoRA is on (lora_rank>0), we only return True for LoRA param.
        param_name: a string like: 'transformer/h/0/attention/wq/kernel'
                    or something JAX derived
        param_value: actual parameter value (unused but needed for named_tree_map)
        We'll just check if it has 'lora_A' or 'lora_B' in it.
        """
        if llama_config.lora_rank > 0:
            # Train only LoRA param
            is_lora = 'lora_A' in param_name or 'lora_B' in param_name
            return is_lora
        else:
            # Full fine-tune
            return True

    # Test trainable_mask with some example parameter names
    if jax.process_index() == 0:
        test_params = [
            'transformer/h/0/attention/wq/kernel',
            'transformer/h/0/attention/wq/lora_A',
            'transformer/h/0/attention/wq/lora_B',
            'transformer/h/0/feed_forward/w1/kernel'
        ]
        logging.info("Testing trainable_mask function:")
        for param in test_params:
            is_trainable = trainable_mask(param)
            logging.info(f"Parameter {param}: {'trainable' if is_trainable else 'frozen'}")

    # Test trainable mask with some example parameters
    if jax.process_index() == 0:
        logging.info("\nTesting trainable_mask before passing to optimizer:")
        test_params = {
            'params': {
                'transformer': {
                    'h': {
                        '0': {
                            'attention': {
                                'wq': {
                                    'kernel': jnp.array([1.0]),
                                    'lora_A': jnp.array([1.0]),
                                    'lora_B': jnp.array([1.0])
                                }
                            }
                        }
                    }
                }
            }
        }
        mask_result = named_tree_map(trainable_mask, test_params, sep='/')
        for path, is_trainable in flatten_dict(mask_result).items():
            logging.info(f"Parameter {'/'.join(str(x) for x in path)}: {'trainable' if is_trainable else 'frozen'}")

    # Use trainable_mask for both weight decay and controlling optimizer state allocation
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        weight_decay_mask=trainable_mask,
        trainable_mask=trainable_mask
    )

    def create_trainstate_from_params(params):
        train_state = TrainState.create(params=params, tx=optimizer, apply_fn=None)
        print("Examining optimizer state:")
        for state in jax.tree_util.tree_leaves(train_state.opt_state):
            print(f"Optimizer state type: {type(state)}")
            print(f"Optimizer state attributes: {dir(state)}")
            print(f"Optimizer state shape: {getattr(state, 'shape', 'no shape')}")
            print(f"Optimizer state sharding: {getattr(state, 'sharding', 'no sharding')}")
            if hasattr(state, 'device_buffers'):
                print(f"Optimizer state device_buffers: {state.device_buffers}")
        return train_state

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        def loss_and_accuracy(params):
            # Apply stop_gradient to non-LoRA params before forward pass
            def maybe_stop_grad(path, p):
                if not trainable_mask(path):
                    return jax.lax.stop_gradient(p)
                return p
            
            params_for_forward = named_tree_map(
                maybe_stop_grad,
                params,
                sep='/'
            )
            
            logits = model.apply(
                params_for_forward, batch['input_tokens'], deterministic=False,
                rngs=rng_generator(LLaMAConfigurator.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(
                logits, batch['target_tokens'], batch['loss_masks']
            )

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        rng = rng_generator()

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

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfigurator.get_partition_rules(), train_state_shapes
    )

    # Log partition specs and actual shapes
    if jax.process_index() == 0:
        logging.info("Examining train state partitioning:")
        # Flatten each field of TrainState separately
        for field in ["params", "opt_state", "step"]:
            logging.info(f"\nExamining {field}:")
            field_partition = getattr(train_state_partition, field)
            field_shapes = getattr(train_state_shapes, field)
            if isinstance(field_partition, (dict, FrozenDict)):
                flat_partition = flatten_dict(field_partition)
                flat_shapes = flatten_dict(field_shapes)
                for name, spec in flat_partition.items():
                    shape = flat_shapes[name].shape if hasattr(flat_shapes[name], 'shape') else None
                    logging.info(f"Parameter {name}:")
                    logging.info(f"  Shape: {shape}")
                    logging.info(f"  Partition spec: {spec}")
            else:
                logging.info(f"  Shape: {getattr(field_shapes, 'shape', None)}")
                logging.info(f"  Partition spec: {field_partition}")

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def remove_frozen_params(tree):
        """
        Build a new tree that sets base-weights to None if LoRA is enabled.
        If lora_rank=0, we return the tree as is (save everything).
        """
        if llama_config.lora_rank == 0:
            return tree  # do nothing

        def maybe_none(param, path):
            # path is a tuple of keys, e.g. ('transformer','h','0','attention','wq','kernel')
            #param_name = '/'.join([str(p) for p in path])
            param_name = str(path)
            if jax.process_index() == 0:
                logging.info(f"Checking parameter: {param_name}")
            is_trainable = trainable_mask(param_name)
            if is_trainable and jax.process_index() == 0:
                logging.info(f"Keeping parameter: {param_name}")
            return param if is_trainable else None

        # We can use named_tree_map for path-based logic:
        # or replicate that logic manually
        pruned = named_tree_map(
            lambda path, leaf: maybe_none(leaf, path),
            tree, sep='/'
        )
        return pruned

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
        hostname = os.uname().nodename
        process_index = jax.process_index()
        logging.info(f"Checkpoint save called on host {hostname} (process {process_index}) at step {step}...")

        # 1. If LoRA is active, prune the base weights from saving
        full_params = train_state.params['params']
        pruned = remove_frozen_params(full_params)
        pruned = prune_none(pruned)

        # 2. Rebuild a partial train_state with pruned params
        partial_state = train_state.replace(params={'params': pruned})

        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        
        checkpoint_dir = os.path.join(logger.output_dir, f"checkpoint_{step}")
        if milestone:
            checkpoint_dir = os.path.join(logger.output_dir, f"milestone_{step}")
        logging.info(f"Saving checkpoint to: {checkpoint_dir}")
        checkpointer.save_all(
            train_state=partial_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )
        logging.info("Checkpoint save complete")
        logging.info("Checkpoint save complete")

    logging.info("Setting up JAX mesh...")
    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    logging.info("JAX mesh initialized")
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )
        logging.info("Loaded checkpoint")

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # For LoRA, we need to initialize LoRA params from scratch
            if llama_config.lora_rank > 0:
                logging.info(f"Initializing LoRA parameters with rank {llama_config.lora_rank}")
                init_state = sharded_init_fn(next_rng())
                # Copy non-LoRA params from checkpoint, keep LoRA params from init
                # restored_params = unfreeze(restored_params)
                init_params = init_state.params
                
                # Debug: print structure before merging
                restored_dict = flatten_dict(restored_params)
                init_dict = flatten_dict(init_params)
                logging.info(f"Restored params has {len(restored_dict)} parameters")
                logging.info(f"Init params has {len(init_dict)} parameters")
                logging.info("Sample of init param paths:")
                for path in list(init_dict.keys())[:5]:
                    logging.info(f"  {path}")

                # unwrapped_restored = unfreeze(restored_params)

                for path, param in flatten_dict(restored_params).items():

                    path_str = str(path)
                    if 'lora_' not in path_str:
                        init_params = set_in_dict(init_params, path, param)
                        if jax.process_index() == 0:
                            logging.info(f"Copied parameter: {path_str}")
                    else:
                        if jax.process_index() == 0:
                            logging.info(f"Skipping LoRA parameter: {path_str}")
            else:
                init_params = restored_params
                # restored_params = freeze(init_params)
            # Create train state with possibly modified params
            train_state = sharded_create_trainstate_from_params(init_params)
            del restored_params

        # Print sharded parameter info on worker 0 only
        if jax.process_index() == 0:
            def print_params_tree(tree, path=''):
                if isinstance(tree, dict):
                    for key, value in tree.items():
                        new_path = f"{path}/{key}" if path else key
                        print_params_tree(value, new_path)
                else:
                    shape_dtype = jax.eval_shape(lambda: tree)
                    logging.info(f"Parameter: {path} with shape {shape_dtype.shape} and sharding {tree.sharding}")

            logging.info("Model parameters after sharding:")
            print_params_tree(train_state.params)

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )

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
