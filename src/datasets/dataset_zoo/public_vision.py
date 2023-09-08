from pandas import DataFrame
from src.datasets.base_dataset_builder import BaseDatasetBuilder
import pytorch_lightning as pl
from src.common.registry import registry
import torchvision.datasets as datasets
from torch.utils.data import random_split
import os 


@registry.register_builder("public_vision")
class PublicVisionDatasetModule(BaseDatasetBuilder):
    def __init__(self, config):
        self.config = config
        self.toy_dataset = self.config.dataset_config.dataset_name
        self.download_bool = self.config.dataset_config.download
        if self.config.dataset_config.save_dir[0] == '/':
            save_dir = self.config.dataset_config.save_dir
        else:
            save_dir = os.path.join(self.config.user_config.data_dir, self.config.dataset_config.save_dir)
        self.save_dir = save_dir

        if ~self.download_bool and self.save_dir is not None:
            print("using the dataset: {}, stored in the following directory: {}. No Download necessary".format(self.toy_dataset, self.save_dir)) 

        elif ~self.download_bool and self.save_dir is None:
            print("<======= warning =======>")
            print("no data is being downloaded, and current data dir is: {}, make sure the correct directory for the dataset has been passed (otherwise only padded black images will be loaded).".format(self.save_dir))
            print("<======= warning =======>")
        
        elif self.download_bool and self.save_dir is not None:
            if ~os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            print("data will be downloaded to the following dir: {}".format(self.save_dir))
        
        else:
            raise RuntimeError('No location specified for saving of public vision dataset')

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
        
        #elif self.toy_dataset=='imagenet':
        #    data_full = datasets.ImageNet(self.save_dir, split='train', transform=transform)
        
        else:
            raise Exception("The following public dataset: {} has not been supported. Please add it to the Toy Vision Data Builder".format(self.toy_dataset))
        
        total_files = len(data_full)
        val_samples = self.config.dataset_config.val_samples
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
            
            #elif self.toy_dataset=='imagenet':
            #    dataset = datasets.ImageNet(self.save_dir, split='val', transform=transform)
        
            else:
                raise Exception("The following public dataset: {} has not been supported. Please add it to the Toy Vision Data Builder".format(self.toy_dataset))

            return dataset


@registry.register_builder("imagenet_vision")
class ImagenetDatasetModule(BaseDatasetBuilder):
    def __init__(self, config):
        self.config= config
        self.toy_dataset = self.config.dataset_config.dataset_name
        self.download_bool = self.config.dataset_config.download
        if self.config.dataset_config.save_dir[0] == '/':
            self.save_dir = self.config.dataset_config.save_dir
        else:
            self.save_dir = os.path.join(self.config.user_config.data_dir, self.config.dataset_config.save_dir)
        
        print("using the dataset: {}, stored in the following directory: {}. No Download necessary".format(self.toy_dataset, self.save_dir)) 
        self.transforms_name = self.config.dataset_config.preprocess.name

    def preprocess(self, split):
        data_transform_cls = registry.get_preprocessor_class(self.transforms_name)
        data_transforms_obj = data_transform_cls(self.config, split)
        return data_transforms_obj

    def data_setup(self, split):
        transform= self.preprocess(split)
        
        if split=='train':
            train_dataset = datasets.ImageNet(self.save_dir, split='train', transform=transform)
            return train_dataset
        
        if split=='val':
            print('<===== Warning! =====>')
            print('do not use val dataset for validation! in Imagenet for val dataset is used for testing')
            print('instead, this code will take a snippet of the training set and use it as val')
            print('<===== Warning! =====>')

            data_full = datasets.ImageNet(self.save_dir, split='train', transform=transform)
            total_files = len(data_full)
            val_samples = self.config.dataset_config.val_samples
            _, val_dataset = random_split(data_full, [total_files-val_samples, val_samples])
            return val_dataset
        
        # Assign test dataset for use in dataloader(s)
        if split == "test":
            print('the val dataset is used in Imagenet for testing')
            test_dataset = datasets.ImageNet(self.save_dir, split='val', transform=transform)
            #datasets.ImageFolder('{}/test/'.format(self.save_dir), transform=transform)
            return test_dataset