import os

os.environ["OMP_NUM_THREADS"] = "8"


from logging import getLogger

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.launcher import LAUNCHER_REGISTRY

logger = getLogger(__name__)


# one of float16, bfloat16
def dtype_resolver(dtype_name):
    return getattr(torch, dtype_name)


def dtype_flag_resolver(dtype_name, compare_to):
    if compare_to == "fp16":
        return dtype_name == "float16"
    elif compare_to == "bf16":
        return dtype_name == "bfloat16"
    raise NotImplementedError()


def accumulation_steps_resolver(per_device_train_batch_size, global_batch_size):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation_steps = global_batch_size // per_device_train_batch_size // world_size
    logger.info(f"Accumulation steps: {accumulation_steps}")
    return accumulation_steps


def subtract_resolver(a, b):
    return int(a) - int(b)


OmegaConf.register_new_resolver("dtype", dtype_resolver)
OmegaConf.register_new_resolver("dtype_flag", dtype_flag_resolver)
OmegaConf.register_new_resolver("accumulation_steps", accumulation_steps_resolver)
OmegaConf.register_new_resolver("debug_report_to", lambda x: "none" if x else "wandb")
OmegaConf.register_new_resolver("subtract", subtract_resolver)


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    if os.environ.get("LOCAL_RANK", "0") == "0":
        if not cfg.debug:
            wandb_config = OmegaConf.to_container(cfg)
            if cfg.wandb_run_name == "auto":
                run_name = "_".join(
                    HydraConfig.get().runtime.output_dir.split("/")[-2:]
                )
            else:
                run_name = cfg.wandb_run_name
            wandb.init(name=run_name, id=run_name, config=wandb_config, resume="allow")

        logger.info(OmegaConf.to_yaml(cfg))

    # Run the launcher
    launcher = LAUNCHER_REGISTRY[cfg.launcher.name]
    launcher = launcher(cfg)
    launcher.run()
    logger.info(f"Run Finished, saved on {cfg.output_dir}")


if __name__ == "__main__":
    main()
