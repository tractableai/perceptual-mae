import yaml
import os 
import importlib
from torch.utils.tensorboard import SummaryWriter
import logging
import warnings
import math
import json
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from omegaconf import OmegaConf
import glob
import time 
from datetime import datetime, timedelta




logger = logging.getLogger(__name__)


def load_yaml(config_path):
    with open(config_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
    return yaml_config

def load_yaml_with_defaults(f):
    default_config = get_default_config_path()
    return OmegaConf.merge(load_yaml(default_config), load_yaml(f))

def get_default_config_path():
    directory = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(directory, "..", "configs")

    # Check for trl defaults
    trl_defaults = os.path.join(configs_dir, "trl_defaults.yaml")
    if os.path.exists(trl_defaults):
        return trl_defaults
    else:
        return os.path.join(configs_dir, "defaults.yaml")


def print_model_parameters(model, return_only=False):
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not return_only:
        logger.info(
            f"Total Parameters: {total_params}. Trained Parameters: {trained_params}"
        )
    return total_params, trained_params

def get_max_updates(config_max_updates, config_max_epochs, train_loader, update_freq):
    if config_max_updates is None and config_max_epochs is None:
        raise ValueError("Neither max_updates nor max_epochs is specified.")

    if isinstance(train_loader, torch.utils.data.IterableDataset):
        warnings.warn(
            "max_epochs not supported for Iterable datasets. Falling back "
            + "to max_updates."
        )
        return config_max_updates, config_max_epochs

    if config_max_updates is not None and config_max_epochs is not None:
        warnings.warn(
            "Both max_updates and max_epochs are specified. "
            + f"Favoring max_epochs: {config_max_epochs}"
        )

    if config_max_epochs is not None:
        assert (
            hasattr(train_loader, "__len__") and len(train_loader) != 0
        ), "max_epochs can't be used with IterableDatasets"
        max_updates = math.ceil(len(train_loader) / update_freq) * config_max_epochs
        max_epochs = config_max_epochs
    else:
        max_updates = config_max_updates
        if hasattr(train_loader, "__len__") and len(train_loader) != 0:
            max_epochs = max_updates / len(train_loader)
        else:
            max_epochs = math.inf

    max_epochs = math.ceil(max_epochs)

    return max_updates, max_epochs

def setup_imports():
    from src.common.registry import registry

    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("corl_root", no_warning=True)

    if root_folder is None:
        root_folder = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.join(root_folder, "..")

        environment_mmf_path = os.environ.get("Core-Research-Library", os.environ.get("SRC_PATH"))
        
        if environment_mmf_path is not None:
            root_folder = environment_mmf_path
        
        registry.register("corl_path", root_folder)
        registry.register("src_path", root_folder)

    trainer_folder = os.path.join(root_folder, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    datasets_folder = os.path.join(root_folder, "datasets")
    datasets_pattern = os.path.join(datasets_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "models")
    losses_folder = os.path.join(root_folder, "losses")
    losses_pattern = os.path.join(losses_folder, "**", "*.py")
    common_folder = os.path.join(root_folder, "common")
    modules_folder = os.path.join(root_folder, "modules")
    model_pattern = os.path.join(model_folder, "**", "*.py")
    common_pattern = os.path.join(common_folder, "**", "*.py")
    modules_pattern = os.path.join(modules_folder, "**", "*.py")


    importlib.import_module("src.common.meter")

    files = (
        glob.glob(datasets_pattern, recursive=True)
        + glob.glob(model_pattern, recursive=True)
        + glob.glob(trainer_pattern, recursive=True)
        + glob.glob(common_pattern, recursive=True)
        + glob.glob(modules_pattern, recursive=True)
        + glob.glob(losses_pattern, recursive=True)
        
    )

    for f in files:
        f = os.path.realpath(f)
        if f.endswith(".py") and not f.endswith("__init__.py"):
            splits = f.split(os.sep)
            import_prefix_index = 0
            for idx, split in enumerate(splits):
                if split == "src":
                    import_prefix_index = idx + 1
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            module = ".".join(["src"] + splits[import_prefix_index:-1] + [module_name])
            importlib.import_module(module)

    registry.register("imports_setup", True)

if __name__=='__main__':
    setup_imports()