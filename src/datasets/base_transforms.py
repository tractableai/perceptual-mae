import collections
import math
import random
import warnings
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import OmegaConf, DictConfig
from torchvision import transforms

"""
The transforms exist in TRL to make data processing pipelines in various
datasets as similar as possible while allowing code reuse.
The transforms also help maintain proper abstractions to keep only what matters
inside the dataset's code. 

This allows us to keep the dataset ``__getitem__``
logic really clean and no need about maintaining opinions about data type.
Processors can work on both images and text due to their generic structure.
To create a new processor, follow these steps:

1. Inherit the ``BaseProcessor`` class.
2. Implement ``__call__`` function which takes in a dict and returns a dict with
   same keys preprocessed as well as any extra keys that need to be returned.
3. Register the processor using ``@registry.register_processor('name')`` to
   registry where 'name' will be used to refer to your processor later.

In processor's config you can specify ``preprocessor`` option to specify
different kind of preprocessors you want in your dataset.
Let's break down processor's config inside a dataset (VQA2.0) a bit to understand
different moving parts.
Config::
    dataset_config:
      vqa2:
        data_dir: ${env.data_dir}
        processors:
          text_processor:
            type: vocab
            params:
              max_length: 14
              vocab:
                type: intersected
                embedding_name: glove.6B.300d
                vocab_file: vqa2/defaults/extras/vocabs/vocabulary_100k.txt
              preprocessor:
                type: simple_sentence
                params: {}
``BaseDataset`` will init the processors and they will available inside your
dataset with same attribute name as the key name, for e.g. `text_processor` will
be available as `self.text_processor` inside your dataset. As is with every module
in MMF, processor also accept a ``DictConfig`` with a `type` and `params`
attributes. `params` defined the custom parameters for each of the processors.
By default, processor initialization process will also init `preprocessor` attribute
which can be a processor config in itself. `preprocessor` can be then be accessed
inside the processor's functions.
Example::
    from mmf.common.registry import registry
    from mmf.datasets.processors import BaseProcessor
    @registry.register_processor('my_processor')
    class MyProcessor(BaseProcessor):
        def __init__(self, config, *args, **kwargs):
            return
        def __call__(self, item, *args, **kwargs):
            text = item['text']
            text = [t.strip() for t in text.split(" ")]
            return {"text": text}
"""


class BaseTransforms:
    """Every processor in TRL needs to inherit this class for compatibility
    with TRL. End user mainly needs to implement ``__call__`` function.
    Args:
        config (DictConfig): Config for this processor, containing `type` and
                             `params` attributes if available.
    """

    def __init__(self, *args, config: Optional[DictConfig] = None, **kwargs):

        #log_class_usage("Processor", self.__class__)
        return

    def __call__(self, item: Any, *args, **kwargs) -> Any:
        """Main function of the processor. Takes in a dict and returns back
        a dict
        Args:
            item (Dict): Some item that needs to be processed.
        Returns:
            Dict: Processed dict.
        """
        return item
