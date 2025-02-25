import time
from functools import partial
import json
import base64
from multiprocessing import Pool
import logging
import os
from tqdm import tqdm
import jax

import mlxu
import numpy as np
from datasets import load_dataset


class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.type = 'huggingface'
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()
        return mlxu.update_config_dict(config, updates)

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        text_processor = TextProcessor(config.text_processor, tokenizer)
        if config.type == 'huggingface':
            return HuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'json':
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.template = """
sequence:
  - no_loss: "{instruction} {input}"
  - with_loss: "{output}"
"""
        config.add_bos_token = True
        config.add_eos_token = True
        config.base64_token_dtype = 'i4'
        config.fields = ''  # Keep for backwards compatibility
        config.fields_from_example = ''  # Keep for backwards compatibility
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        self.tokenizer = tokenizer

        # Pre-tokenize template if provided
        self.cached_segments = None
        if self.config.template:
            if jax.process_index() == 0:
                logging.info("Starting template pre-tokenization...")
            import yaml
            template = yaml.safe_load(self.config.template)
            # Convert \n to actual newlines in template
            for segment in template['sequence']:
                for key, content in segment.items():
                    segment[key] = content.replace('\\n', '\n')
            self.cached_segments = []

            # Process each segment in the sequence
            for segment in template['sequence']:
                # Each segment is a dict with one key-value pair
                for loss_key, content in segment.items():
                    # Set loss mask based on key
                    mask = 0.0 if loss_key == 'no_loss' else 1.0

                    # Find all dynamic field references
                    import re
                    field_pattern = r'\{([^}]+)\}'
                    parts = re.split(field_pattern, content)

                    # Pre-tokenize static parts
                    cached_parts = []
                    for i, part in enumerate(parts):
                        if i % 2 == 0:  # Static content
                            tokens = []
                            text = part
                            # Handle special tokens
                            while '<|' in text and '|>' in text:
                                start = text.index('<|')
                                end = text.index('|>') + 2

                                # Add text before special token
                                if start > 0:
                                    prefix_tokens = self.tokenizer.encode(
                                        text[:start], add_special_tokens=False
                                    )
                                    tokens.extend(prefix_tokens)

                                # Add special token
                                token_name = text[start + 2:end - 2]
                                if token_name == 'bos':
                                    tokens.append(self.tokenizer.bos_token_id)
                                elif token_name == 'eos':
                                    tokens.append(self.tokenizer.eos_token_id)
                                else:
                                    special_token = f"<|{token_name}|>"
                                    token_id = self.tokenizer.convert_tokens_to_ids(special_token)
                                    tokens.append(token_id)

                                text = text[end:]

                            # Add remaining text
                            if text:
                                tokens.extend(self.tokenizer.encode(text, add_special_tokens=False))

                            cached_parts.append(('static', tokens))
                        else:  # Dynamic field name
                            cached_parts.append(('field', part))

                    self.cached_segments.append((mask, cached_parts))
            if jax.process_index() == 0:
                logging.info("Template pre-tokenization complete")

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        token_buffer = []
        loss_mask_buffer = []

        if self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.cached_segments is not None:
            # Use pre-tokenized segments
            for mask, parts in self.cached_segments:
                for part_type, part in parts:
                    if part_type == 'static':
                        # Add pre-tokenized static content
                        token_buffer.extend(part)
                        loss_mask_buffer.extend([mask] * len(part))
                    else:  # Dynamic field
                        # Tokenize and add dynamic content
                        field_content = str(example.get(part, ''))
                        tokens = self.tokenizer.encode(field_content, add_special_tokens=False)
                        token_buffer.extend(tokens)
                        loss_mask_buffer.extend([mask] * len(tokens))
        else:
            # Fallback to old template processing
            import yaml
            template = yaml.safe_load(self.config.template)
            for segment in template['sequence']:
                for loss_key, content in segment.items():
                    mask = 0.0 if loss_key == 'no_loss' else 1.0
                    text = content.format(**example)
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask] * len(tokens))

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, *aux


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.batch_token_dtype = 'i4'
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor

        # Create cache directory if needed
        cache_dir = os.path.expanduser("~/.cache/easylm/datasets")
        # Create full path including dataset-specific subdirectories
        cache_path = os.path.join(cache_dir, f"{self.config.path}_{name}_{split}_{tokenizer.name_or_path}.npz")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Cache path was already created above

        if os.path.exists(cache_path):
            logging.info(f"Loading preprocessed dataset from cache: {cache_path}")
            cached_data = np.load(cache_path, allow_pickle=True)
            self._dataset = list(cached_data['dataset'])  # Convert to list directly
            logging.info(f"Loaded {len(self._dataset)} examples from cache")
        else:
            logging.info(f"Loading dataset from {self.config.path}")
            raw_dataset = load_dataset(
                self.config.path, name, split=split, streaming=self.config.streaming
            )

            # Process all examples
            processed_examples = []
            total_examples = len(raw_dataset)
            if jax.process_index() == 0:
                logging.info(f"Starting dataset pre-tokenization of {total_examples} examples...")
            for i, example in enumerate(tqdm(raw_dataset, desc="Processing dataset")):
                if i % 1000 == 0 and jax.process_index() == 0:
                    logging.info(f"Pre-tokenized {i}/{total_examples} examples...")
                tokens, loss_masks = text_processor(example)
                processed_examples.append({
                    'tokens': np.array(tokens, dtype=np.int32),
                    'loss_masks': np.array(loss_masks, dtype=np.float32)
                })
            if jax.process_index() == 0:
                logging.info(f"Dataset pre-tokenization complete. Processed {total_examples} examples.")
            self._dataset = processed_examples

            # Save to cache
            logging.info(f"Saving processed dataset to cache: {cache_path}")
            np.savez(cache_path, dataset=self._dataset)
            logging.info(f"Processed and cached {len(self._dataset)} examples")

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                # Use pre-tokenized data directly
                token_buffer.extend(example['tokens'])
                loss_mask_buffer.extend(example['loss_masks'])
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    batch = {
                        'input_tokens': np.array(token_buffer[:chunk_size],
                                                 dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'target_tokens': np.array(token_buffer[1:chunk_size + 1],
                                                  dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    if self.config.always_start_with_bos:
                        batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                    yield batch, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(mlxu.ConfigDict(state_dict['config']))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:  # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index
                self._index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(), self.config.tokenizer_parallel_batch_size
            )
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, loc, index in self.parallel_example_iterator():
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (
                        (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                batch = {
                    'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                if self.config.always_start_with_bos:
                    batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(mlxu.ConfigDict(state_dict['config']))
        self._index = state_dict.get('index', self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens', self.config.tokens_count_at_start)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)
