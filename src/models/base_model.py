from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch
"""
To build a new model inherit ``BaseModel`` class and adhere to
a fixed format. 

1. Inherit ``BaseModel`` class, make sure to call ``super().__init__()`` in your
   class's ``__init__`` function.
2. Implement `build` function for your model. If you build everything in ``__init__``,
   you can just return in this function.
3. Write a `forward` function which returns a dict.
4. Register using ``@registry.register_model("key")`` decorator on top of the
   class.

If you are doing logits based predictions, the dict you return from your model
should contain a `scores` field. Losses are automatically calculated by the
``BaseModel`` class and added to this dict if not present.

Example::
    import torch
    from src.common.registry import registry
    from src.models.base_model import BaseModel
    
    @registry.register("model_name")
    class model_name(BaseModel):
        # config is model_config from global config
        def __init__(self, config):
            super().__init__(config)
        def build(self):
            ....
        def forward(self, sample_list):
            scores = torch.rand(sample_list.get_batch_size(), 3127)
            return {"scores": scores}
"""


class BaseModel(pl.LightningModule):
    """For integration with the trainer, datasets and other features,
    models needs to inherit this class, call `super`, write a build function,
    write a forward function taking a ``SampleList`` as input and returning a
    dict as output and finally, register it using ``@registry.register_model``
    """
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self):
        """Warning: this is just empty shell for code implemented in other class.
        Configure and set forward propagation of models here.
        """
        raise NotImplementedError
        
    @abstractmethod
    def training_step(self):
        """Warning: this is just empty shell for code implemented in other class.
        Configure and set the training strategy for single batch here
        """
        raise NotImplementedError
    
    @abstractmethod
    def validation_step(self):
        """Warning: this is just empty shell for code implemented in other class.
        Configure and set the validation strategy for single batch here
        """
        raise NotImplementedError

    #@abstractmethod
    def test_step(self):
        """Warning: this is just empty shell for code implemented in other class.
        Configure and set testing strategy for single batch here.
        """
        raise NotImplementedError

    
    def calculate_loss(self, pred, target):
        loss_list=[]
        for loss_name, _ in self.loss_fnc.items():
            loss_val = self.loss_fnc[loss_name](pred, target) #[self.loss_names[0]]
            loss_list.append(loss_val)
        loss= sum(loss_list)

        return loss

    def calculate_metrics(self, pred, target):
        metric_output={}
        for metric_name, _ in self.metrics.items():
            metric_val = self.metrics[metric_name](pred, target)
            metric_output[metric_name]=metric_val

        return metric_output
    
    #def get_optimizer_parameters(self):
    #    """Warning: this is just empty shell for code implemented in other class.
    #    If you don't want to train all parameters in the model i.e. some parts of the models
    #    are frozen; then define which parameters you'd like the optimizer to optimize here.
    #    """
    #    raise NotImplementedError