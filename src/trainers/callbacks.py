from src.common.registry import registry
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pathlib import Path
import os 

@registry.register_callback('CheckpointEveryNSteps')
class CheckpointEveryNSteps(Callback):
    """
    Save a checkpoint every N steps (batch iterations), 
    instead of / in addition to; Lightning's default 
    that checkpoints based on validation loss.
    """

    def __init__(self, config):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.config = config
        callback_args = self.filter_callback(self.config.training.callbacks)

        self.save_step_frequency = callback_args.params.save_step_frequency
        self.prefix = callback_args.params.prefix
        self.use_modelcheckpoint_filename = callback_args.params.use_modelcheckpoint_filename
    
    def filter_callback(self, callback_list):
        for i in callback_list:
            if i['type']=='CheckpointEveryNSteps':
                return i 
            else:
                pass

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=:03d}_{global_step=:04d}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            #print(ckpt_path)
            trainer.save_checkpoint(ckpt_path)

