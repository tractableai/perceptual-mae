import torch 
import pytorch_lightning as pl
from abc import ABC, abstractmethod

from src.common.registry import registry
#from utils.logger import log_class_usage
from omegaconf import DictConfig


@registry.register_trainer("base")
class BaseTrainer(ABC):
    def __init__(self):
        pass

    def configure_device(self):
        """Warning: this is just empty shell for code implemented in other class.
        Configure and set device properties here.
        """
        raise NotImplementedError

    def configure_seed(self):
        """Configure seed and related changes like torch deterministic etc shere.
        Warning: Empty shell for code to be implemented in other class.
        """
        raise NotImplementedError

    def configure_callbacks(self):
        """Configure callbacks and add callbacks be executed during
        different events during training, validation or test.
        Warning: Empty shell for code to be implemented in other class.
        """
        raise NotImplementedError

    def load_datasets(self):
        """Loads datasets and dataloaders.
        Warning: Empty shell for code to be implemented in other class.
        """
        raise NotImplementedError

    def load_model(self):
        """Load the model.
        Warning: Empty shell for code to be implemented in other class.
        """
        raise NotImplementedError

    def load_metrics(self):
        """Load metrics for evaluation.
        Warning: Empty shell for code to be implemented in other class.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """Runs full training and optimization.
        Warning: Empty shell for code to be implemented in other class.
        """
        raise NotImplementedError

    @abstractmethod
    def val(self):
        """Runs inference and validation, generate predictions.
        Warning: Empty shell for code to be implemented in other class.
        """
        raise NotImplementedError