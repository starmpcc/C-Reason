import re
from logging import getLogger

from datasets import concatenate_datasets
from trl import GRPOConfig, GRPOTrainer

from . import register_launcher
from .base import Launcher

logger = getLogger(__name__)


@register_launcher("grpo")
class GRPOLauncher(Launcher):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_dataset = concatenate_datasets(list(self.train_datasets.values()))
        self.train_dataset = self.train_dataset.shuffle(seed=42)

        grpo_config = GRPOConfig(
            **cfg.launcher.grpo_arguments,
            **cfg.launcher.training_arguments,
        )

        if self.lora_args is not None:
            logger.warning(
                "GRPO does not support LoRA. Ignoring the LoRA arguments provided."
            )
        self.trainer = GRPOTrainer(
            self.model,
            args=grpo_config,
            reward_funcs=self.reward_function,
            train_dataset=self.train_dataset,
        )
        # Remove "None" bos_token_id
        if self.tokenizer.bos_token_id == None:
            self.trainer.train_dataset = self.trainer.train_dataset.map(
                lambda x: {
                    k: v[1:]
                    for k, v in x.items()
                    if isinstance(v, list) and "answer" not in k
                }
            )
        return

    def reward_function(self, completions, label_0, **kwargs):
        # How can I check the genreated value without additional generation?
        matches = [
            re.search(r"\\boxed\{([A-E])\}", completion) for completion in completions
        ]
        contents = [match.group(1) if match else "" for match in matches]
        rewards = [float(content == label) for content, label in zip(contents, label_0)]
        return rewards
