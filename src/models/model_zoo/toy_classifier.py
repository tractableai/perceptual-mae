import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.models.base_model import BaseModel
from src.models.modules.layer_utils import build_image_encoder
from src.common.registry import registry
from src.utils.builder import build_metrics
from torchmetrics import Accuracy, Precision, F1Score, Recall, AUROC

@registry.register_model("toy_classifier")
class ToyClassifier(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = self.config.model_config.layers
        self.classifier_params = self.config.model_config.classifier.params
        self.encoder_arch= self.config.model_config.image_encoder.name
        self.pretrained = self.config.model_config.image_encoder.pretrained

        self.resnet = build_image_encoder(arch=self.encoder_arch, pretrained=self.pretrained)
        
        # Number of input channels B x C x H x W
        if self.layers.input_dim!=3:
            self.firstconv = nn.Conv2d(self.layers.input_dim, 3,
                                kernel_size=(7, 7), stride=(2, 2),
                                padding=(3, 3), bias=False)
        
        self.resnet.fc = nn.Linear(self.classifier_params.in_dim, self.classifier_params.logits)
    
        if self.pretrained!=True:
            # if not loading pre-trained weights; use xavier initialiser
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
        
        self.loss= nn.CrossEntropyLoss()
        #self.metrics = #build_metrics(self.config)
        #self.metrics = #build_metrics(self.config)
        self.Precision = Precision(num_classes=10, average='samples', multiclass=True)
        self.Recall = Recall(num_classes=10, average='samples', multiclass=True)
        self.F1 = F1Score(num_classes=10, average='samples', mdmc_average='samplewise', multiclass=True)
        #self.Auroc= AUROC(num_classes=10, average='micro')
        self.Accuracy = Accuracy(num_classes=10, average='samples', multiclass=True)

    def forward(self, x):
        if self.layers.input_dim!=3:
            x= self.firstconv(x)
        
        x= self.resnet(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # calculate metrics;
        preds = torch.argmax(logits.clone(), dim=1)


        f1_score = self.F1(preds, y)  
        precision_score = self.Precision(preds, y) 
        recall_score = self.Recall(preds, y) 
        accuracy_score = self.Accuracy(preds, y) 
        #auroc_score = self.Auroc(preds, y) 

        self.log('train_{}'.format('f1'), f1_score, rank_zero_only=True)
        self.log('train_{}'.format('precision'), precision_score, rank_zero_only=True)
        self.log('train_{}'.format('recall'), recall_score, rank_zero_only=True)
        self.log('train_{}'.format('acurracy'), accuracy_score, rank_zero_only=True)
        #self.log('train_{}'.format('auroc'), auroc_score, rank_zero_only=True)

        #for metric_name,_ in self.metrics.items():
            #metric_val = self.metrics[metric_name](logits, y)
        #    self.log('train_{}'.format(metric_name), self.metrics[metric_name](logits, y))

        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        #preds = out.clone()

        f1_score = self.F1(preds, y)  
        precision_score = self.Precision(preds, y) 
        recall_score = self.Recall(preds, y) 
        accuracy_score = self.Accuracy(preds, y) 
        #auroc_score = self.Auroc(preds, y) 

        self.log('val_{}'.format('f1'), f1_score, rank_zero_only=True)
        self.log('val_{}'.format('precision'), precision_score, rank_zero_only=True)
        self.log('val_{}'.format('recall'), recall_score, rank_zero_only=True)
        self.log('val_{}'.format('acurracy'), accuracy_score, rank_zero_only=True)
        #self.log('train_{}'.format('auroc'), auroc_score, rank_zero_only=True)
        
        #for metric_name,_ in self.metrics.items():
            #metric_val = self.metrics[metric_name](preds, y)
        #    self.log('val_{}'.format(metric_name), self.metrics[metric_name](preds, y))

        # Calling self.log will surface up scalars for you in TensorBoard
        #self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        #self.log("val_acc", self.accuracy, prog_bar=True)
        return {'val_loss': loss}
        
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
        
    
    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9) #build_optimizer(self, self.config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer) #, step_size=30, gamma=0.9) #build_scheduler(optimizer, self.config)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
