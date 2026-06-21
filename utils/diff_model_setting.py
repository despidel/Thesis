from __future__ import annotations

from argparse import Namespace
import json
import logging
import sys

import torch
import torch.distributed as dist
from monai.utils import RankFilter


def setup_logging(logger_name: str = "") -> logging.Logger:
    """
    Setup the logging configuration.

    Args:
        logger_name (str): logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    # Configure basic logging with basicConfig
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,  # stdout instead of default stderr
    )

    # Get the logger
    logger = logging.getLogger(logger_name)

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    if dist.is_initialized():
        logger.addFilter(RankFilter())

    # Set the level
    logger.setLevel(logging.INFO)

    return logger


def load_configs(args: Namespace, config_paths):
    """Load JSON configs and merge them into args namespace."""
    if isinstance(config_paths, str):
        config_paths = [config_paths]
    
    for config_path in config_paths:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        for k, v in config_data.items():
            setattr(args, k, v)

    # Ensure img_filename is always a dict, not a string
    if hasattr(args, "img_filename") and isinstance(args.img_filename, str):
        args.img_filename = json.loads(args.img_filename)
    
    return args


def initialize_distributed(num_gpus: int) -> tuple:
    """
    Initialize distributed training.

    Returns:
        tuple: local_rank, world_size, and device.
    """
    if torch.cuda.is_available() and num_gpus > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return local_rank, world_size, device
