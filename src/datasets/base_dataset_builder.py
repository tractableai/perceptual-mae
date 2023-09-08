import torch
from torch.utils.data.dataset import Dataset
import pytorch_lightning as pl
from src.common.registry import registry
from typing import Optional
import numpy as np 
import pandas as pd
""" 
In TRL, for adding new datasets, dataset builder for datasets needs to be
added. A new dataset builder must inherit ``BaseDatasetBuilder`` class which
follows similar structure to Pytorch Lightning data module, enabling the user
to train with either their own custom trainer and or Pytorch Lightning trainer

Example::
    from torch.utils.data import Dataset
    from src.datasets.base_dataset_builder import BaseDatasetBuilder
    from src.common.registry import registry
    
    @registry.register_builder("my")
    class MyBuilder(BaseDatasetBuilder):
        def __init__(self):
            super().__init__("my")
        
        def setup(self, split):
            ...
            return Dataset()

"""

class BaseDatasetBuilder(pl.LightningDataModule):
    
    def __init__(self, config):
        super().__init__()
        self.config= config
        self.dataset_name = self.config.dataset_config.dataset_name
        self.transforms_name = self.config.dataset_config.preprocess.type
        
        self.num_gpus = self.get_num_gpus()
   
    def get_num_gpus(self):
        if self.config.trainer.params.gpus==-1:
            num_gpus = len(torch.cuda.device_count())
        elif type(self.config.self.config.trainer.params.gpus)==list:
            num_gpus= len(self.config.self.config.trainer.params.gpus)
        return num_gpus

    def preprocess(self, split):
        """
        Initiate your pre-processing Transforms class
        Args:
            split = str, defining whether the transforms is for 'train', 'val' or 'test'
        Returns:
            the transformations class
        """
        data_transform_cls = registry.get_transforms_class(self.transforms_name)
        data_transforms_obj = data_transform_cls(self.config, split)
        return data_transforms_obj


    def prepare_data(self):
        """Warning: this is just empty shell for code implemented in other class (Optional)
        This is to define any data downloads (such as using sample Torchvision data, tokenize, etc ...)
        """
        raise NotImplementedError

    def data_setup(self, split: Optional[str] = None):
        """Initiate the dataset here, with the relevant split,
        the splits can be 'train', 'val' or 'test'
        This is what you change in your builder!
        """
        dataset_cls= registry.get_dataset_class(self.dataset_name)
        transforms = self.preprocess(split)
        dataset_obj = dataset_cls(self.config, split=split, transforms=transforms)

        return dataset_obj

    def train_dataloader(self):
        train_dataset = self.data_setup('train')
        print('total Train sample #: {}'.format(len(train_dataset)))
        return torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=int(self.config.training.batch_size), 
                                           shuffle=True, 
                                           num_workers=int(self.config.training.num_workers),
                                           pin_memory=True, 
                                           drop_last=True)
    
    def val_dataloader(self):
        val_dataset = self.data_setup('val')
        print('total Validation sample #: {}'.format(len(val_dataset)))
        return torch.utils.data.DataLoader(val_dataset, 
                                           batch_size=int(self.config.training.batch_size), 
                                           shuffle= False, 
                                           num_workers=int(self.config.training.num_workers),
                                           pin_memory=True, 
                                           drop_last=True)

    def test_dataloader(self):
        test_dataset = self.data_setup('test')
        print('total Testing sample #: {}'.format(len(test_dataset)))
        return torch.utils.data.DataLoader(test_dataset, 
                                           batch_size=int(self.config.training.batch_size), 
                                           shuffle= False, 
                                           num_workers=int(self.config.training.num_workers),
                                           pin_memory=True, 
                                           drop_last=True)

    # --------- Helper functions ----------
    def get_label_list(self, label_cols):
        """ 
        converts the label columns args (misture of str and dict) into a list of str.
        args:
            label_cols: a list of columns to be selected for labels (typically fed in as a mixture of str and dict)
        output:
            list: converts it into a single list
        """
        
        new_list=[]
        for i in label_cols:
            if type(i)== str:
                new_list.append(i)
            elif type(i) == dict:
                for _, v in i.items():
                    new_list.append(v)
        return new_list
        
    def get_usable_columns_with_defaults(self, label_cols):
        label_cols = self.get_label_list(label_cols)
        usecols= self.DEFAULT_USE_COLS + label_cols
        return usecols

    def read_df(self, path, cols):
        if path.endswith('.csv'):
            if cols!=None:
                return pd.read_csv(path, usecols=lambda s: s in cols)
            else:
                return pd.read_csv(path)
        else:
            if cols!=None:
                return pd.read_parquet(path, columns= cols)
            else:
                return pd.read_parquet(path)

