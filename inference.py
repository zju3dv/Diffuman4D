from typing import Any, Dict, Optional, Tuple

import os
import hydra
from glob import glob
from omegaconf import DictConfig

from src.samplers.sampling_runner import SamplingRunner
from src.utils import RankedLogger, print_config_tree

log = RankedLogger(__name__, rank_zero_only=True)

# for easier debugging
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def inference(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating pipelines <{cfg.model._target_}>")
    pipelines = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating sampler <{cfg.sampler._target_}>")
    sampler = hydra.utils.instantiate(cfg.sampler, dataset=dataset, pipelines=pipelines)
    runner = SamplingRunner(sampler)

    if cfg.sampling:
        log.info("Sampling...")
        runner.inference()

    if cfg.to_nerfstudio:
        log.info("Converting results to nerfstudio format...")
        runner.to_nerfstudio()

    if cfg.evaluating:
        log.info("Evaluating results...")
        runner.evaluate()


@hydra.main(version_base="1.3", config_path="configs", config_name="test.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point"""
    print_config_tree(cfg, resolve=True, save_to_file=True)

    inference(cfg)


if __name__ == "__main__":
    main()
