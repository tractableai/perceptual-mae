from pandas import DataFrame
from src.datasets.base_dataset_builder import BaseDatasetBuilder
import pytorch_lightning as pl
from src.common.registry import registry
import torchvision.datasets as datasets
from torch.utils.data import random_split
import os 


@registry.register_builder("toy_vision")
class ToyVisionDatasetModule(BaseDatasetBuilder):
    def __init__(self, config):
        self.config= config
        self.toy_dataset = self.config.dataset_config.dataset_name
        if self.config.dataset_config.save_dir[0] == '/':
            save_dir = self.config.dataset_config.save_dir
        else:
            save_dir = os.path.join(self.config.user_config.data_dir, self.config.dataset_config.save_dir)
        self.save_dir = save_dir
        if ~os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.transforms_name = self.config.dataset_config.preprocess.name

    def preprocess(self, split):
        data_transform_cls = registry.get_preprocessor_class(self.transforms_name)
        data_transforms_obj = data_transform_cls(self.config, split)
        return data_transforms_obj

    def data_setup(self, split):
        transform= self.preprocess(split)
        if self.toy_dataset=='cifar':
            data_full = datasets.CIFAR100(self.save_dir, train=True, download=True, transform=transform)
        
        elif self.toy_dataset=='mnist':
            data_full = datasets.MNIST(self.save_dir, train=True, download=True, transform=transform)
        
        elif self.toy_dataset=='imagenet':
            data_full = datasets.ImageNet(self.save_dir, split='train', transform=transform)
        
        else:
            raise Exception("The following toy dataset: {} has not been supported. Please add it to the Toy Vision Data Builder".format(self.toy_dataset))
        
        total_files = len(data_full)
        val_samples = 500
        train_dataset, val_dataset = random_split(data_full, [total_files-val_samples, val_samples])
        
        if split=='train':
            return train_dataset
        if split=='val':
            return val_dataset
        
        # Assign test dataset for use in dataloader(s)
        if split == "test":
            if self.toy_dataset=='cifar':
                dataset = datasets.CIFAR100(self.save_dir, train=False, transform=transform)
            
            elif self.toy_dataset=='mnist':
                dataset = datasets.MNIST(self.save_dir, train=False, transform=transform)
            
            elif self.toy_dataset=='imagenet':
                dataset = datasets.ImageNet(self.save_dir, split='val', transform=transform)
        
            else:
                raise Exception("The following toy dataset: {} has not been supported. Please add it to the Toy Vision Data Builder".format(self.toy_dataset))

            return dataset