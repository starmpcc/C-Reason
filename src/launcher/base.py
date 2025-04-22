import gc
import os
from logging import getLogger
from shutil import rmtree
from tempfile import _get_candidate_names

import torch
import wandb
from accelerate.utils import set_seed
from liger_kernel.transformers import (
    apply_liger_kernel_to_llama,
    apply_liger_kernel_to_mixtral,
    apply_liger_kernel_to_phi3,
    apply_liger_kernel_to_qwen2,
)
from omegaconf import ListConfig
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from ..dataset import get_datasets
from ..utils import answer_phrase, get_accuracy
from . import register_launcher

apply_liger_kernel_to_llama(cross_entropy=True, fused_linear_cross_entropy=False)
apply_liger_kernel_to_mixtral(cross_entropy=True, fused_linear_cross_entropy=False)
apply_liger_kernel_to_qwen2(cross_entropy=True, fused_linear_cross_entropy=False)
apply_liger_kernel_to_phi3(cross_entropy=True, fused_linear_cross_entropy=False)

logger = getLogger(__name__)
set_seed(42)


class Launcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lora_args = None
        if os.path.exists(cfg.model_path) and "adapter_config.json" in os.listdir(
            cfg.model_path
        ):
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                **cfg.model,
                is_trainable=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(**cfg.model)
            if cfg.use_lora:
                # ListConfig -> List
                lora_args = dict(cfg.launcher.lora_args)
                lora_args = {
                    k: list(v) if isinstance(v, ListConfig) else v
                    for k, v in lora_args.items()
                }
                self.lora_args = LoraConfig(**lora_args)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path,
            cache_dir=cfg.model.cache_dir,
        )

        self.tokenizer.model_max_length = cfg.launcher.max_length

        self.had_padding_adjusted = False

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

            if hasattr(self.model.model, "base_model"):  # PEFTModel
                embed_tokens = self.model.model.base_model.embed_tokens
            else:
                embed_tokens = self.model.model.embed_tokens
            embed_tokens.padding_idx = self.tokenizer.pad_token_id
            embed_tokens._fill_padding_idx_with_zero()
            self.had_padding_adjusted = True

        self.tokenizer.padding_side = "left"

        self.training_args = TrainingArguments(**cfg.launcher.training_arguments)

        self.train_datasets = get_datasets(cfg, self.tokenizer)

    def evaluate(self):
        raise NotImplementedError()

    def run(self):
        if getattr(self.trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy

            fsdp_plugin = self.trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(self.trainer.model)

        self.trainer.train()
        self.tokenizer.add_eos_token = False  # Recover original setting
        if self.trainer.is_fsdp_enabled:
            self.trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
                "FULL_STATE_DICT"
            )
        self.trainer.save_model(self.cfg.output_dir)

        if self.trainer.accelerator.is_main_process:
            if self.trainer.is_fsdp_enabled:
                if self.cfg.use_lora:
                    model = AutoPeftModelForCausalLM.from_pretrained(
                        self.cfg.output_dir,
                        device_map="cpu",
                        cache_dir=self.cfg.model.cache_dir,
                        trust_remote_code=True,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.cfg.output_dir,
                        device_map="cpu",
                        cache_dir=self.cfg.model.cache_dir,
                        trust_remote_code=True,
                    )
            else:
                model = self.trainer.model

            if self.had_padding_adjusted:
                model.resize_token_embeddings(len(self.tokenizer) - 1)
                model.config.pad_token_id = 0

            if not self.trainer.is_deepspeed_enabled:
                model.save_pretrained(self.cfg.output_dir)

            tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_path, cache_dir=self.cfg.model.cache_dir
            )
            tokenizer.save_pretrained(self.cfg.output_dir)


@register_launcher("vllm")
class VLLMLauncher:
    def __init__(self, cfg):
        # Do not call super().__init__(cfg). Initializing model init CUDA -> MP error
        self.cfg = cfg
        self.model = None
        tokenizer_path = (
            "microsoft/phi-4"
            if self.cfg.model_path == "OpenMeditron/Meditron3-Phi4-14B"
            else self.cfg.model_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, cache_dir=self.cfg.model.cache_dir
        )
        self.datasets = get_datasets(cfg, self.tokenizer)

        self.temp_dir = None

        tp_size = torch.cuda.device_count()
        if self.tokenizer.name_or_path == "microsoft/phi-4" and tp_size >= 2:
            tp_size = 2

        # if model is lora:
        # Path can be a hf repo
        if os.path.exists(cfg.model_path) and "adapter_config.json" in os.listdir(
            cfg.model_path
        ):
            model = AutoPeftModelForCausalLM.from_pretrained(
                cfg.model_path,
                device_map="auto",
                torch_dtype="auto",
                cache_dir=cfg.model.cache_dir,
            )
            model = model.merge_and_unload()
            self.temp_dir = os.path.join("/dev/shm", next(_get_candidate_names()))
            os.makedirs(self.temp_dir)
            model.save_pretrained(self.temp_dir)
            self.tokenizer.save_pretrained(self.temp_dir)
            self.cfg.model_path = self.temp_dir

            if (
                model.peft_config["default"].base_model_name_or_path
                == "microsoft/phi-4"
            ):
                tp_size = 2
            del model
            gc.collect()
            torch.cuda.empty_cache()

        self.sampling_params = SamplingParams(
            **self.cfg.launcher.sampling_params,
        )

        self.model = LLM(tensor_parallel_size=tp_size, **self.cfg.launcher.llm)

    def generate(self, batch, dset_name):
        outputs = self.model.generate(
            prompts=batch["prompt"],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        # NOTE: only support n=1
        batch["generated"] = [output.outputs[0].text for output in outputs]
        # Check finish_reason
        batch["truncated"] = [
            output.outputs[0].finish_reason == "length" for output in outputs
        ]
        # Extract Value
        num_questions = self.cfg.dataset.get(dset_name).num_questions
        for i in range(num_questions):
            batch[f"value_{i}"] = self.extract_values(
                batch["prompt"], batch["generated"], i
            )

        return batch

    def get_logits_processor(self, dset_name, idx=None):
        if idx is not None:
            num_choices = self.cfg.dataset.get(dset_name).get("num_choices_" + str(idx))
        else:
            num_choices = self.cfg.dataset.get(dset_name).num_choices
        choices_tokens = [
            self.tokenizer.encode(chr(65 + i), add_special_tokens=False)[0]
            for i in range(num_choices)
        ]
        vocab_size = self.model.llm_engine.model_config.get_vocab_size()
        choices_tensor = torch.zeros(vocab_size, device="cuda")
        choices_tensor[choices_tokens] = 100

        def _logits_processor(input_ids, scores):
            return scores + choices_tensor

        return _logits_processor

    def extract_values(self, prompts, generateds, idx=-1):
        _answer_phrase = (
            answer_phrase if idx == -1 else f"Therefore, the answer to Q{idx+1} is "
        )
        input_tokens = []
        answer_phrase_tokens = self.tokenizer.encode(_answer_phrase)
        max_model_len = self.model.llm_engine.get_model_config().max_model_len
        for prompt, generated in zip(prompts, generateds):
            tokens = self.tokenizer.encode(prompt + generated)
            if len(tokens) + len(answer_phrase_tokens) + 1 >= max_model_len:
                tokens = tokens[: max_model_len - len(answer_phrase_tokens) - 1]
            input_tokens.append(
                TokensPrompt(prompt_token_ids=tokens + answer_phrase_tokens)
            )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logits_processors=[self.logits_processors[idx]],
        )
        value_outputs = self.model.generate(
            prompts=input_tokens,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        values = [output.outputs[0].text.strip() for output in value_outputs]

        return values

    def run(self):
        logging_dict = {}
        for k, v in self.datasets.items():
            num_questions = self.cfg.dataset.get(k).num_questions
            self.logits_processors = [
                self.get_logits_processor(k, i) for i in range(num_questions)
            ]
            dset = v.map(
                self.generate,
                batched=True,
                batch_size=100,
                fn_kwargs={"dset_name": k},
            )
            dset.save_to_disk(os.path.join(self.cfg.output_dir, k))

            # Print Evaluation Results & Log
            for i in range(num_questions):
                acc = get_accuracy(dset[f"value_{i}"], dset[f"label_{i}"])
                logging_dict[f"{k}_{self.cfg.launcher.train_type}_acc_{i}"] = acc
                logger.info(f"{k}_{self.cfg.launcher.train_type}_acc_{i}: {acc}")

        logger.info(str(logging_dict))
        if not self.cfg.debug:
            for k, v in logging_dict.items():
                wandb.run.summary[k] = v
        if self.temp_dir is not None:
            rmtree(self.temp_dir, ignore_errors=True)
