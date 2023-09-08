# import torch
# from torch.utils.data.dataset import Dataset
# from PIL import Image
# import pandas as pd
# import pandas
# import numpy as np 
# import random
# from awsio.python.lib.io.s3.s3dataset import S3Dataset
# import io
# import _pywrap_s3_io
# from src.common.registry import registry

# #@registry.register_dataset("base")
# class BaseDataset(S3Dataset, Dataset):
#     def __init__(self, dataset_name, config, dataset_type="train",transforms=None):
#         """
#         Base class for implementing a dataset. Inherits from PyTorch's Dataset class
#         but adds some custom functionality on top. 
#         Args:
#         dataset_name (str): Name of your dataset to be used a representative
#             in text strings
#         dataset_type (str): Type of your dataset. Normally, train|val|test
#         config (DictConfig): Configuration for the current dataset
#         """
#         self.config= config
#         self._dataset_name = dataset_name
#         self._dataset_type = dataset_type
#         self._global_config = registry.get("config")
        
#     def __getitem__(self, idx):
#         """
#         Basically, __getitem__ of a torch dataset.
#         Args:
#             idx (int): Index of the sample to be loaded.
#         """

#         raise NotImplementedError 

#     def __len__(self):
#         """
#         Basically, __len__ of a torch dataset.
#         returns:
#                 length of the data
#         """

#         raise NotImplementedError

#     def visualize(self, num_samples=1, *args, **kwargs):
#         raise NotImplementedError(
#             f"{self.dataset_name} doesn't implement visualize function"
#         )

#     @property
#     def dataset_type(self):
#         return self._dataset_type

#     @property
#     def name(self):
#         return self._dataset_name

#     @property
#     def dataset_name(self):
#         return self._dataset_name

#     @dataset_name.setter
#     def dataset_name(self, name):
#         self._dataset_name = name

#     @dataset_type.setter
#     def dataset_type(self, dataset_type):
#         self._dataset_type = dataset_type
