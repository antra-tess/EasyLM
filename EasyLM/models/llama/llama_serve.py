import pprint
from functools import partial
import time
import logging

import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import optax
from transformers import (
    AutoTokenizer, GenerationConfig, FlaxLogitsProcessorList
)

from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.serving import LMServer
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    with_sharding_constraint, FlaxTemperatureLogitsWarper
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfigurator, FlaxLLaMAForCausalLM
)


from EasyLM.models.llama.llama_config import create_llama_flags

FLAGS, FLAGS_DEF = create_llama_flags()


class ModelServer(LMServer):

    def __init__(self, config):
        super().__init__(config)


        logging.info("Initializing JAX distributed configuration...")
        JaxDistributedConfig.initialize(FLAGS.jax_distributed)
        set_random_seed(FLAGS.seed)

        logging.info("Loading tokenizers...")
        tokenizer_start = time.time()
        self.prefix_tokenizer = AutoTokenizer.from_pretrained(
            FLAGS.tokenizer, truncation_side='left', padding_side='left'
        )
        self.prefix_tokenizer.pad_token = self.prefix_tokenizer.eos_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            FLAGS.tokenizer, truncation_side='right', padding_side='right'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)

        logging.info("Loading model checkpoint and initializing model...")
        model_start = time.time()
        with jax.default_device(jax.devices("cpu")[0]):
            logging.info(f"Loading checkpoint from {FLAGS.load_checkpoint}")
            # Create model to get parameter shapes
            hf_model = FlaxLLaMAForCausalLM(
                llama_config,
                input_shape=(1, FLAGS.seq_length),
                seed=FLAGS.seed,
                _do_init=False,
                dtype=get_float_dtype_by_name(FLAGS.dtype),
                param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
            )

            # Get base parameter partition rules
            model_ps = match_partition_rules(
                LLaMAConfigurator.get_base_param_rules(), hf_model.params_shape_tree
            )
            shard_fns, gather_fns = make_shard_and_gather_fns(
                model_ps, get_float_dtype_by_name(FLAGS.param_dtype)
            )

            # Load checkpoint with sharding functions
            _, params = StreamingCheckpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint,
                disallow_trainstate=True,
                trainstate_shard_fns={'params': shard_fns}  # Single wrap for base_params mode
            )

        model_ps = match_partition_rules(
            LLaMAConfigurator.get_partition_rules(), params
        )
        shard_fns, _ = make_shard_and_gather_fns(
            model_ps, get_float_dtype_by_name(FLAGS.param_dtype)
        )

        @partial(
            pjit,
            in_shardings=(model_ps, PS(), PS()),
            out_shardings=(PS(), PS(), PS())
        )
        def forward_loglikelihood(params, rng, batch):
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
            rng_generator = JaxRNG(rng)
            input_tokens = batch['input_tokens']
            output_tokens = batch['output_tokens']
            input_mask = batch['input_mask']
            output_mask = batch['output_mask']

            logits = hf_model.module.apply(
                params, input_tokens, attention_mask=input_mask,
                deterministic=True, rngs=rng_generator(LLaMAConfigurator.rng_keys()),
            ).logits
            loglikelihood = -optax.softmax_cross_entropy_with_integer_labels(
                logits, output_tokens
            )
            loglikelihood = jnp.sum(loglikelihood * output_mask, axis=-1)
            match_count = jnp.sum(
                (jnp.argmax(logits, axis=-1) == output_tokens) * output_mask,
                axis=-1
            )
            total = jnp.sum(output_mask, axis=-1)
            is_greedy = match_count == total
            return loglikelihood, is_greedy, rng_generator()

        self.forward_loglikelihood = forward_loglikelihood

        @partial(
            pjit,
            in_shardings=(model_ps, PS(), PS(), PS()),
            out_shardings=(PS(), PS())
        )
        def forward_generate(params, rng, batch, temperature):
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
            rng_generator = JaxRNG(rng)
            output = hf_model.generate(
                batch['input_tokens'],
                attention_mask=batch['attention_mask'],
                params=params['params'],
                prng_key=rng_generator(),
                logits_processor=FlaxLogitsProcessorList(
                    [FlaxTemperatureLogitsWarper(temperature)]
                ),
                generation_config=GenerationConfig(
                    max_new_tokens=FLAGS.seq_length - FLAGS.input_length,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=FLAGS.do_sample,
                    num_beams=FLAGS.num_beams,
                    top_k=FLAGS.top_k,
                    top_p=FLAGS.top_p,
                )
            ).sequences[:, batch['input_tokens'].shape[1]:]
            return output, rng_generator()
        self.forward_generate = forward_generate

        @partial(
            pjit,
            in_shardings=(model_ps, PS(), PS()),
            out_shardings=(PS(), PS())
        )
        def forward_greedy_generate(params, rng, batch):
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
            rng_generator = JaxRNG(rng)
            output = hf_model.generate(
                batch['input_tokens'],
                attention_mask=batch['attention_mask'],
                params=params['params'],
                prng_key=rng_generator(),
                generation_config=GenerationConfig(
                    max_new_tokens=FLAGS.seq_length - FLAGS.input_length,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    num_beams=1,
                )
            ).sequences[:, batch['input_tokens'].shape[1]:]
            return output, rng_generator()
        self.forward_greedy_generate = forward_greedy_generate

        logging.info("Setting up JAX mesh and compiling serving functions...")
        mesh_start = time.time()
        self.mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
        with self.mesh:
            logging.info("Sharding parameters across mesh...")
            self.params = tree_apply(shard_fns, params)
            self.sharded_rng = next_rng()
            logging.info(f"Mesh setup complete. Took {time.time() - mesh_start:.1f}s")

    def loglikelihood(self, prefix_text, text):
        prefix = self.prefix_tokenizer(
            prefix_text,
            padding='max_length',
            truncation=True,
            max_length=FLAGS.input_length,
            return_tensors='np',
        )
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=FLAGS.seq_length - FLAGS.input_length,
            return_tensors='np',
        )
        output_tokens = np.concatenate([prefix.input_ids, inputs.input_ids], axis=1)
        bos_tokens = np.full(
            (output_tokens.shape[0], 1), self.tokenizer.bos_token_id, dtype=np.int32
        )
        input_tokens = np.concatenate([bos_tokens, output_tokens[:, :-1]], axis=-1)
        input_mask = np.concatenate(
            [prefix.attention_mask, inputs.attention_mask], axis=1
        )
        if FLAGS.add_bos_token:
            bos_mask = np.ones_like(input_mask[:, :1])
        else:
            bos_mask = np.zeros_like(input_mask[:, :1])

        input_mask = np.concatenate([bos_mask, input_mask[:, :-1]], axis=1)
        output_mask = np.concatenate(
            [np.zeros_like(prefix.attention_mask), inputs.attention_mask], axis=1
        )
        batch = dict(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_mask=input_mask,
            output_mask=output_mask,
        )
        with self.mesh:
            loglikelihood, is_greedy, sharded_rng = self.forward_loglikelihood(
                self.params, self.sharded_rng, batch
            )
            loglikelihood, is_greedy = jax.device_get((loglikelihood, is_greedy))
        return loglikelihood, is_greedy

    def loglikelihood_rolling(self, text):
        inputs = self.tokenizer(
            text,
            padding='longest',
            truncation=False,
            max_length=np.iinfo(np.int32).max,
            return_tensors='np',
        )
        batch_size = inputs.input_ids.shape[0]
        output_tokens = inputs.input_ids
        attention_mask = inputs.attention_mask

        if output_tokens.shape[1] < FLAGS.seq_length:
            padding_length = FLAGS.seq_length - output_tokens.shape[1]
            pad_tokens = np.full(
                (batch_size, padding_length), self.tokenizer.pad_token_id, dtype=np.int32
            )
            output_tokens = np.concatenate([output_tokens, pad_tokens], axis=-1)
            pad_mask = np.zeros(
                (batch_size, padding_length), dtype=inputs.attention_mask.dtype
            )
            attention_mask = np.concatenate([attention_mask, pad_mask], axis=-1)

        bos_tokens = np.full(
            (batch_size, 1), self.tokenizer.bos_token_id, dtype=np.int32
        )
        input_tokens = np.concatenate([bos_tokens, output_tokens[:, :-1]], axis=-1)
        bos_mask = np.ones((batch_size, 1), dtype=inputs.attention_mask.dtype)
        total_seq_length = output_tokens.shape[1]

        total_loglikelihood = 0.0
        total_is_greedy = True
        # Sliding window
        for i in range(0, total_seq_length, FLAGS.seq_length):
            # Last window
            if i + FLAGS.seq_length > total_seq_length:
                last_output_mask = np.copy(attention_mask[:, -FLAGS.seq_length:])
                last_output_mask[:, :i - total_seq_length] = 0.0

                batch = dict(
                    input_tokens=input_tokens[:, -FLAGS.seq_length:],
                    output_tokens=output_tokens[:, -FLAGS.seq_length:],
                    input_mask=attention_mask[:, -FLAGS.seq_length:],
                    output_mask=last_output_mask,
                )

            # Normal window
            else:
                batch = dict(
                    input_tokens=input_tokens[:, i:i + FLAGS.seq_length],
                    output_tokens=output_tokens[:, i:i + FLAGS.seq_length],
                    input_mask=attention_mask[:, i:i + FLAGS.seq_length],
                    output_mask=attention_mask[:, i:i + FLAGS.seq_length],
                )

            with self.mesh:
                loglikelihood, is_greedy, sharded_rng = self.forward_loglikelihood(
                    self.params, self.sharded_rng, batch
                )
                loglikelihood, is_greedy = jax.device_get((loglikelihood, is_greedy))

            total_loglikelihood += loglikelihood
            total_is_greedy = np.logical_and(is_greedy, total_is_greedy)

        return total_loglikelihood, total_is_greedy

    def generate(self, text, temperature):
        if jax.process_index() == 0:
            logging.info(f"Generating with text: {text}")
            logging.info(f"Temperature: {temperature}")
        inputs = self.prefix_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=FLAGS.input_length,
            return_tensors='np',
        )
        if jax.process_index() == 0:
            logging.info(f"Input tokens shape: {inputs.input_ids.shape}")
        input_tokens = inputs.input_ids
        input_mask = inputs.attention_mask
        if FLAGS.add_bos_token:
            input_tokens[:, 0] = self.tokenizer.bos_token_id
            input_mask[:, 0] = 1
        batch = dict(
            input_tokens=input_tokens,
            attention_mask=input_mask,
        )
        with self.mesh:
            output, sharded_rng = self.forward_generate(
                self.params, self.sharded_rng, batch, temperature
            )
            output = jax.device_get(output)
        output_text = []
        for text in list(self.tokenizer.batch_decode(output)):
            if self.tokenizer.eos_token in text:
                text = text.split(self.tokenizer.eos_token, maxsplit=1)[0]
            output_text.append(text)

        return output_text

    def greedy_until(self, prefix_text, until, max_length):
        all_outputs = []
        for pf, ut in zip(prefix_text, until):
            if isinstance(ut, str):
                ut = [ut]
            total_length = 0
            total_generated = ''

            while total_length < max_length:
                pf_tokens = self.tokenizer(
                    pf,
                    padding=False,
                    truncation=False,
                    max_length=np.iinfo(np.int32).max,
                    return_tensors='np',
                )
                input_tokens = pf_tokens.input_ids
                attention_mask = pf_tokens.attention_mask

                if input_tokens.shape[1] < FLAGS.input_length:
                    extra = FLAGS.input_length - input_tokens.shape[1]
                    pad_tokens = np.full(
                        (1, extra), self.tokenizer.pad_token_id, dtype=np.int32
                    )
                    input_tokens = np.concatenate(
                        [pad_tokens, input_tokens], axis=1
                    )
                    pad_attention = np.zeros((1, extra), dtype=attention_mask.dtype)
                    attention_mask = np.concatenate(
                        [pad_attention, attention_mask], axis=1
                    )
                elif input_tokens.shape[1] > FLAGS.input_length:
                    input_tokens = input_tokens[:, -FLAGS.input_length:]
                    attention_mask = attention_mask[:, -FLAGS.input_length:]

                if FLAGS.add_bos_token:
                    input_tokens[:, 0] = self.tokenizer.bos_token_id
                    attention_mask[:, 0] = 1

                batch = dict(input_tokens=input_tokens, attention_mask=attention_mask)

                with self.mesh:
                    output, self.sharded_rng = self.forward_greedy_generate(
                        self.params, self.sharded_rng, batch
                    )
                    output = jax.device_get(output)

                total_length += output.shape[1]
                output_text = self.tokenizer.batch_decode(output)[0]
                total_generated = total_generated + output_text
                pf = pf + output_text

                done = False
                for s in ut:
                    if s in total_generated:
                        total_generated = total_generated.split(s, maxsplit=1)[0]
                        done = True
                if done:
                    break

            all_outputs.append(total_generated)

        return all_outputs


def main(argv):
    logging.info("Starting LLaMA serving initialization...")
    start_time = time.time()

    logging.info("Initializing model server...")
    server = ModelServer(FLAGS.lm_server)
    logging.info(f"Total initialization time: {time.time() - start_time:.1f}s")
    logging.info("Starting server...")
    server.run()


if __name__ == "__main__":
    mlxu.run(main)
