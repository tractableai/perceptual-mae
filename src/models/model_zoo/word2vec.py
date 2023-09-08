import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.models.base_model import BaseModel
from src.common.registry import registry
from src.utils.builder import build_metrics, build_loss
import torchmetrics 

@registry.register_model("skip_gram")
class SkipGram_Model(BaseModel):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        # Embedding
        self.embeddings = nn.Embedding(self.model_config.vocab_size, self.model_config.embedding_dim)
        # Linerar
        self.linear1 = nn.Linear(self.model_config.context_size * self.model_config.embedding_dim, 128)
        self.linear2 = nn.Linear(128, self.model_config.vocab_size)


        self.loss= build_loss(self.config) # should be NLLLoss()
        self.metrics = build_metrics(self.config)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        embeds = self.embeddings(x).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss(outputs, y)
        for metric_name,_ in self.metrics.items():
            #metric_val = self.metrics[metric_name](logits, y)
            self.log('train_{}'.format(metric_name), self.metrics[metric_name](logits, y))

        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = F.nll_loss(outputs, y)
        preds = torch.argmax(logits, dim=1)
        
        for metric_name,_ in self.metrics.items():
            #metric_val = self.metrics[metric_name](preds, y)
            self.log('val_{}'.format(metric_name), self.metrics[metric_name](preds, y))

        # Calling self.log will surface up scalars for you in TensorBoard
        self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return {'val_loss': loss}
        
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
        

@registry.register_model("cbow")
class CBOW_Model(BaseModel):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        # Embedding
        self.embeddings = nn.Embedding(self.model_config.vocab_size, self.model_config.embedding_dim)
        # Linerar
        self.linear = nn.Linear(self.model_config.embedding_dim, self.model_config.vocab_size)

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss(outputs, y)
        for metric_name,_ in self.metrics.items():
            #metric_val = self.metrics[metric_name](logits, y)
            self.log('train_{}'.format(metric_name), self.metrics[metric_name](logits, y))

        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = F.nll_loss(outputs, y)
        preds = torch.argmax(logits, dim=1)
        
        for metric_name,_ in self.metrics.items():
            #metric_val = self.metrics[metric_name](preds, y)
            self.log('val_{}'.format(metric_name), self.metrics[metric_name](preds, y))

        # Calling self.log will surface up scalars for you in TensorBoard
        self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return {'val_loss': loss}
        
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)