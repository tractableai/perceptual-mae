from src.models.modules.layer_utils import * 
import torch 
import torch.nn as nn 


class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super(LanguageEncoder, self).__init__()
        self.config = config
        self.model_config = self.config.model_config.language_encoder
        self.pretrained = self.model_config.pretrained

        # initiate hugging face model;
        self.encoder = build_language_encoder(self.model_config, arch=self.model_config.name, pretrained=self.pretrained)

    def forward(self, x):
        output = self.encoder(**x)
        final_output= output.last_hidden_state

        return final_output


