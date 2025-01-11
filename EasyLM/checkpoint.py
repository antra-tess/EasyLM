import os
import logging
import numpy as np
from ml_collections import ConfigDict
import mlxu
import jax
import jax.numpy as jnp
import flax
from flax.serialization import (
    from_bytes, to_bytes, to_state_dict, from_state_dict
)
from flax.traverse_util import flatten_dict, unflatten_dict, empty_node
import msgpack

from EasyLM.jax_utils import tree_apply, float_tensor_to_dtype


class StreamingCheckpointer(object):
    """ Custom msgpack checkpointer that saves large train states by serializing
        and saving tensors one by one in a streaming fashion. Avoids running
        out of memory or local TPU disk with default flax checkpointer.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.float_dtype = 'bf16'
        config.save_optimizer_state = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, checkpoint_dir, enable=True):
        self.config = self.get_default_config(config)
        self.checkpoint_dir = checkpoint_dir
        self.enable = enable

    def save_checkpoint(self, train_state, filename, gather_fns=None):
        if self.enable:
            path = os.path.join(self.checkpoint_dir, filename)
        else:
            path = '/dev/null'
        self.save_train_state_to_file(
            train_state, path, gather_fns, self.config.float_dtype
        )

    @staticmethod
    def save_train_state_to_file(train_state, path, gather_fns=None, float_dtype=None):
        train_state = to_state_dict(train_state)
        packer = msgpack.Packer()
        flattend_train_state = flatten_dict(train_state)
        if gather_fns is not None:
            gather_fns = flatten_dict(to_state_dict(gather_fns))

        with mlxu.open_file(path, "wb") as fout:
            for key, value in flattend_train_state.items():
                if gather_fns is not None:
                    value = gather_fns[key](value)
                value = float_tensor_to_dtype(value, float_dtype)
                fout.write(packer.pack((key, to_bytes(value))))

    def save_pickle(self, obj, filename):
        if self.enable:
            path = os.path.join(self.checkpoint_dir, filename)
        else:
            path = '/dev/null'
        mlxu.save_pickle(obj, path)

    def save_all(self, train_state, gather_fns, metadata=None, dataset=None, milestone=False, params_wrapped=True):
        step = int(jax.device_get(train_state.step))
        if self.config.save_optimizer_state:
            checkpoint_state = train_state
            checkpoint_name = 'streaming_train_state'
            checkpoint_gather_fns = gather_fns
            if not params_wrapped:
                raise ValueError("save_all with optimizer state requires params_wrapped=True")
        else:
            checkpoint_state = train_state.params['params']
            checkpoint_name = 'streaming_params'
            if params_wrapped:
                checkpoint_gather_fns = gather_fns.params['params']
            else:
                checkpoint_gather_fns = gather_fns.params

        # Strip any prefix like "params::" from checkpoint directory path
        base_dir = self.checkpoint_dir.split("::")[-1] if "::" in self.checkpoint_dir else self.checkpoint_dir
        
        if milestone:
            # Save a milestone checkpoint that will not be overwritten
            checkpoint_dir = os.path.join(base_dir, f"milestone_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.save_pickle(metadata, os.path.join(checkpoint_dir, 'metadata.pkl'))
            self.save_pickle(dataset, os.path.join(checkpoint_dir, 'dataset.pkl'))
            self.save_checkpoint(
                checkpoint_state, os.path.join(checkpoint_dir, checkpoint_name), checkpoint_gather_fns
            )
        else:
            # Save a normal checkpoint that can be overwritten
            checkpoint_dir = os.path.join(base_dir, f"checkpoint_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.save_pickle(metadata, os.path.join(checkpoint_dir, 'metadata.pkl'))
            self.save_pickle(dataset, os.path.join(checkpoint_dir, 'dataset.pkl'))
            self.save_checkpoint(
                checkpoint_state, os.path.join(checkpoint_dir, checkpoint_name), checkpoint_gather_fns
            )

    @staticmethod
    def load_checkpoint(path, target=None, shard_fns=None, remove_dict_prefix=None, restore_state=True, require_sharding=True):
        if jax.process_index() == 0:
            logging.info(f"Loading checkpoint from {path}")
            if shard_fns is not None:
                logging.info("Shard functions provided:")
                for k, v in flatten_dict(to_state_dict(shard_fns)).items():
                    logging.info(f"  {'/'.join(str(x) for x in k)}")
            if remove_dict_prefix is not None:
                logging.info(f"Removing prefix: {remove_dict_prefix}")
        if target is not None:
            if jax.process_index() == 0:
                # Get first parameter's dtype
                first_param = jax.tree_util.tree_leaves(target)[0]
                logging.info(f"First parameter dtype during load checkpoint: {first_param.dtype}")


        if shard_fns is not None:
            shard_fns = flatten_dict(
                to_state_dict(shard_fns)
            )
        if remove_dict_prefix is not None:
            remove_dict_prefix = tuple(remove_dict_prefix)
        flattend_train_state = {}
        counter = 0
        with mlxu.open_file(path) as fin:
            # Get file size for progress bar
            fin.seek(0, 2)  # Seek to end
            total_size = fin.tell()
            fin.seek(0)  # Back to start

            # 83886080 bytes = 80 MB, which is 16 blocks on GCS
            unpacker = msgpack.Unpacker(fin, read_size=83886080, max_buffer_size=0)
            if jax.process_index() == 0:
                from tqdm import tqdm
                pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Loading checkpoint")
            for key, value in unpacker:
                key = tuple(key)
                if remove_dict_prefix is not None:
                    if key[:len(remove_dict_prefix)] == remove_dict_prefix:
                        key = key[len(remove_dict_prefix):]
                    else:
                        continue

                tensor = from_bytes(None, value)
                if jax.process_index() == 0 and 'lora_' in '/'.join(str(x) for x in key):
                    logging.info(f"Loaded LoRA tensor {'/'.join(str(x) for x in key)}:")
                    logging.info(f"  Shape: {tensor.shape}")
                    logging.info(f"  Mean: {jnp.mean(tensor)}")
                    logging.info(f"  First 10 values: {tensor.flatten()[:10]}")
                if shard_fns is not None:
                    counter += 1
                    try:
                        tensor = shard_fns[key](tensor)
                    except KeyError as e:
                        if jax.process_index() == 0:
                            logging.info(f"No sharding function found for key {key}")
                            logging.info(f"Available shard_fns keys: {list(shard_fns.keys())}")
                        raise
                if jax.process_index() == 0:
                    from tqdm import tqdm
                    pbar = tqdm(total=len(flattend_train_state), desc="Loading checkpoint")
                    pbar.update(1)
                flattend_train_state[key] = tensor
                if jax.process_index() == 0:
                    pbar.update(unpacker.tell() - pbar.n)  # Update to current position
                if jax.process_index() == 0:
                    pbar.update(1)
                if jax.process_index() == 0:
                    pbar.update(1)
        if require_sharding and counter == 0:
            raise ValueError(f"No tensor sharding was applied {path}")


        # if target is not None:
        #     flattened_target = flatten_dict(
        #         to_state_dict(target), keep_empty_nodes=True
        #     )
        #     for key, value in flattened_target.items():
        #         if 'lora_' in str(key):
        #             # Initialize LoRA parameters with zeros
        #             # Handle both actual arrays and ShapeDtypeStruct
        #             if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        #                 flattend_train_state[key] = jnp.zeros(value.shape, dtype=value.dtype)
        #             else:
        #                 pass
        #             continue
        #         if key not in flattend_train_state and value is empty_node:
        #             flattend_train_state[key] = value
        #
        # else:
        #     #logging.info(f"target is None, not initializing LoRA parameters")
        #     pass

        train_state = unflatten_dict(flattend_train_state)

        if target is None or not restore_state:
            if jax.process_index() == 0:
                logging.info("Loaded state without restoring target state")
            return train_state

        # Create a copy of train_state with all target keys
        full_state = {}
        flattened_target = flatten_dict(to_state_dict(target))
        flattened_state = flatten_dict(train_state)
        
        # Copy all available keys from train_state
        counter = 0
        kept = 0
        for key in flattened_target.keys():
            if key in flattened_state:
                full_state[key] = flattened_state[key]
                if jax.process_index() == 0:
                    logging.info(f"Restored key {key}")
                counter += 1
            else:
                # For missing keys (like lm_head in LoRA), use target's value
                full_state[key] = flattened_target[key]
                if jax.process_index() == 0:
                    logging.info(f"Kept key {key} from target")
                kept += 1
        if jax.process_index() == 0:
            logging.info(f"Restored {counter} keys from train_state, kept {kept} keys from target")
        
        return from_state_dict(target, unflatten_dict(full_state))

    @staticmethod
    def load_flax_checkpoint(path, target=None, shard_fns=None):
        """ Load a standard flax checkpoint that's not saved with the
            msgpack streaming format.
        """
        with mlxu.open_file(path, "rb") as fin:
            encoded_bytes = fin.read()

        state_dict = flax.serialization.msgpack_restore(encoded_bytes)
        if shard_fns is not None:
            shard_fns = to_state_dict(shard_fns)
            state_dict = tree_apply(shard_fns, state_dict)

        if target is None:
            return state_dict
        return from_state_dict(target, state_dict)

    @classmethod
    def load_trainstate_checkpoint(cls, load_from, trainstate_target=None,
                                   trainstate_shard_fns=None,
                                   disallow_trainstate=False):

        if trainstate_shard_fns is not None:
            #print("trainstate_shard_fns: ", trainstate_shard_fns)
            # Handle both dictionary and object cases
            if isinstance(trainstate_shard_fns, dict):
                params_shard_fns = trainstate_shard_fns['params']
            else:
                params_shard_fns = trainstate_shard_fns.params['params']
        else:
            params_shard_fns = None

        try:
            load_type, load_path = load_from.split('::', 1)
        except ValueError:
            raise ValueError(f'Invalid load_from format: {load_from}')
        if disallow_trainstate:
            assert load_type != 'trainstate', 'Loading full trainstate is not allowed!'
        train_state = None
        restored_params = None
        if load_type == 'trainstate':
            # Load the entire train state in the streaming format
            train_state = cls.load_checkpoint(
                path=load_path,
                target=trainstate_target,
                shard_fns=trainstate_shard_fns,
            )
        elif load_type == 'trainstate_params':
            if trainstate_target is not None:
                params_target = trainstate_target.params['params']
            else:
                params_target = None
            # Load the params part of the train state in the streaming format
            restored_params = cls.load_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,
                remove_dict_prefix=('params', 'params'),
            )
            restored_params = {'params': restored_params}
        elif load_type == 'params':
            if trainstate_target is not None:
                params_target = trainstate_target.params['params']
            else:
                params_target = None
            # Load the params in the streaming format
            restored_params = cls.load_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,
                restore_state=True,
            )
            restored_params = {'params': restored_params}
        elif load_type == 'base_params':
            # Load base model parameters only, no train state
            if trainstate_target is not None:
                params_target = trainstate_target.get('params', None)
            else:
                params_target = None

            restored_params = cls.load_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,  # Use params sharding directly
                restore_state=True
            )
            # Wrap in params dict structure
            if not isinstance(restored_params, dict) or 'params' not in restored_params:
                restored_params = {'params': restored_params}
            return restored_params  # Return only params, no train state
        elif load_type == 'base_params_unsharded':
            # Load base model parameters without requiring sharding functions
            if trainstate_target is not None:
                params_target = trainstate_target.get('params', None)
            else:
                params_target = None

            restored_params = cls.load_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=None,  # No sharding required
                restore_state=True,
                require_sharding=False  # Skip sharding requirement check
            )
            # Wrap in params dict structure
            if not isinstance(restored_params, dict) or 'params' not in restored_params:
                restored_params = {'params': restored_params}
            return None, restored_params  # Return only params, no train state
        elif load_type == 'flax_params':
            if trainstate_target is not None:
                params_target = trainstate_target.params['params']
            else:
                params_target = None
            # Load the params in the standard flax format (non-streaming)
            # This requires the entire params to fit in memory
            restored_params = cls.load_flax_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns
            )
            restored_params = {'params': restored_params}
        else:
            raise ValueError(f'Invalid load_from type: {load_type}')

        return train_state, restored_params
