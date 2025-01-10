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

jax.config.update("jax_compilation_cache_dir", "/mnt/disk2/jax_cache")
#jax.config.update("jax_explain_cache_misses", True)
#jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

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
        keep_recent=50,  # Number of recent checkpoints to keep
    ),
)

def logginginfo(msg, *args):
    if jax.process_index() == 0:
        logging.info(msg, *args)


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
        FLAGS.optimizer, weight_decay_mask=lambda x: True
    )
    logginginfo(f"Optimizer setup complete: {str(optimizer_info)}, {str(optimizer)}, {str(type(optimizer))}")

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        logginginfo("Initializing model parameters...")
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        )
        logginginfo("Model parameter initialization complete")
        logginginfo("Creating train state from parameters...")
        train_state = TrainState.create(params=params, tx=optimizer, apply_fn=None)
        logginginfo("Train state creation complete")
        return train_state

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        def loss_and_accuracy(params):

            logits = model.apply(
                params, batch['input_tokens'], deterministic=False,
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

    logginginfo("Calculating partitioning...")
    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    logginginfo("Train state shape calculation complete")
    train_state_partition = match_partition_rules(
        LLaMAConfigurator.get_partition_rules(), train_state_shapes
    )
    logginginfo("Train state partitioning complete")

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
    logginginfo("Partitioned functions created")


    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        hostname = os.uname().nodename
        process_index = jax.process_index()
        logginginfo(f"Checkpoint save called on host {hostname} (process {process_index}) at step {step}...")
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
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )
        logginginfo("Checkpoint save complete")
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
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )
            logginginfo("Loaded checkpoint")
            log_memory_usage("After checkpoint load")

        if train_state is None and restored_params is not None:
            train_state = sharded_create_trainstate_from_params(restored_params)

        if train_state is None:
            logginginfo("Initializing parameters from scratch")
            log_memory_usage("Before initialization")
            train_state = sharded_init_fn(next_rng())
            log_memory_usage("After initialization")
            logginginfo("Initialization complete")

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

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        log_memory_usage("Before training loop")
        logginginfo("Starting training loop...")
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
