import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.registry import registry
from src.losses.image_reconstruction import MaskedImageLoss
"""
Potential Language-vision losses;
https://arxiv.org/pdf/2111.02387.pdf

MLM: masked language modelling
MIM: masked image modelling 
ITM: image-text matching
WRA: word-region alginment
ITC: image-text contrastive learning.
"""


class MaskedLanguageVisionLoss(nn.Module):
    def __init__(self, config, patch_embed):
        super().__init__()
        self.config = config 
        self.patch_embed = patch_embed
        # MLM, MIM & ITM Loss
        self.mlm_loss= MLM(self.config)
        self.mim_loss= MaskedImageLoss(self.config, self.patch_embed)
        self.itm_loss= ITM(self.config)

    def forward(self, pred, gt):
        vision_pred, text_pred, itm_class_pred = pred
        vision_gt, text_gt, itm_class_gt = gt 
        losses={}
        losses['mim']= self.mim_loss(vision_pred, vision_gt)
        losses['mlm']= self.mlm_loss(text_pred, text_gt)
        losses['itm']= self.itm_loss(itm_class_pred, itm_class_gt)
        return losses



@registry.register_loss('mlm')
class MLMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config= config
        self.vocab_size = self.config.model_config.language_encoder.vocab_size
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, pred, gt):
        mlm_loss = self.cross_entropy(pred.view(-1, self.vocab_size),
                                      gt.view(-1))
        return mlm_loss

    
@registry.register_loss('itm')
class ITMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss() #build_loss(self.config)

    def forward(self, pred, gt):
        return self.bce_loss(pred, gt)
