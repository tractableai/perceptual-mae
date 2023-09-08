import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.models.base_model import BaseModel
from src.models.modules.layer_utils import *
from src.models.modules.output_heads import *
from src.common.registry import registry
from src.utils.builder import build_loss, build_metrics, build_optimizer, build_scheduler
from torchmetrics import WordErrorRate, MatchErrorRate, Accuracy
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


#from sklearn.metrics import f1, precision_score, recall_score, roc_auc_score, accuracy_score, average_precision_score

@registry.register_model("masked_language_model")
class MaskedLanguageModel(BaseModel):
    """
    Test > HuggingFace AutoModelForMaskedLM > tokens classification
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.model_params = self.model_config.language_encoder

        print('using {} model as language encoder'.format(self.model_config.language_encoder.name))
        
        # -------- Language Encoder --------;
        # arch='roberta-base', task='mlm', pretrained=True
        self.model, lang_config= build_language_encoder(arch=self.model_params.name,
                                                   task=self.model_params.task,
                                                   pretrained=self.model_params.pretrained, 
                                                   lang_model_config= None) 
        print('model details: \n {}'.format(config))

        if self.model_params.task!= 'mlm' or self.model_params.task!='masked_language_modelling':
            # assuming above returned a pre-trained roberta
            self.mlm_head = RobertaMLMHead(lang_config)
            # ------- build loss --------
            self.loss_fnc = nn.CrossEntropyLoss()
        
        # ---- build metrics -----
        
        self.perplexity= Perplexity()
        
        # NO need to build loss function; handled in the AutoModelForMaskedLM model class within huggingface.
        # unless you are defining a custom head!

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        """
        loss handled within the model, output = dictionary
        output= ['loss', 'logits', 'hidden_states', 'attentions']
        check: https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.MaskedLMOutput
        for more info
        """
        input_ids, attention_mask, labels = batch['masked_inputs'], batch['attention_mask'], batch['labels']
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs[0], outputs[1]
        perplexity = torch.exp(loss)
        # clone logits for metrics (don't want gradients to pass)
        self.log('train_loss', loss, rank_zero_only=True, prog_bar=True)
        #self.log('train_accuracy', acc, rank_zero_only=True, prog_bar=True)
        self.log('train_perplexity', perplexity, rank_zero_only=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['masked_inputs'], batch['attention_mask'], batch['labels']
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        val_loss, logits = outputs[:2]

        if self.model_config.num_labels >= 1:
            # predicting more than one masked label;
            preds = torch.argmax(logits, axis=1)
        
        elif self.model_config.num_labels == 1:
            preds = logits.squeeze()

        perplexity=torch.exp(val_loss)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", val_loss, prog_bar=True, rank_zero_only=True)
        self.log('val_perplexity', perplexity, rank_zero_only=True, prog_bar=True)
        
        # to use tensorboard summarywriter use; self.experiment

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    
    def configure_optimizers(self):
        """
        Prepare optimizer and schedule (linear warmup and decay)
        """
        optimizer = AdamW(self.parameters(), lr= float(self.config.optimizer.params.learning_rate), eps=float(self.config.optimizer.params.eps))

        self.caclulate_training_steps(stage="fit")
        print(self.total_steps, 'total no. of steps for scheduler')

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # default value in run_glue.py 
                                                    num_training_steps = self.total_steps)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def caclulate_training_steps(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        #train_loader = self.train_loader
        # Calculate total steps
        tb_size = self.config.training.batch_size * max(1, len(self.trainer.gpus))
        ab_size = self.trainer.accumulate_grad_batches * 40 #float(self.trainer.max_epochs)
        self.total_steps = (77523 // tb_size) // ab_size


    def get_output_embeddings(self):
        return self.mlm_head.decoder



class Perplexity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, output, target):
        ce = self.cross_entropy(output, target)
        perplexity = torch.exp(ce)
        return perplexity


