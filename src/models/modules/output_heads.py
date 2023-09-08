import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from src.models.modules.layer_utils import get_attn_config

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    """Image Text Matching Head (Binary Classification)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.itm_combine_multi_image = self.model_config.itm_combine_multi_image
        # if multi image input; cls feats for all images needs to be compressed into single dim vector.
        if self.itm_combine_multi_image:
            self.compress_ = nn.Conv2d(5,1,1)
        self.fc = nn.Linear(self.model_config.hidden_size * 2, 1)

    def forward(self, x):
        if self.itm_combine_multi_image:
            x = self.compress_(x.unsqueeze(2))
            x=x.squeeze()
        x = self.fc(x)
        return x


class RobertaMLMHead(nn.Module):
    """Roberta Head for masked language modeling.
    from https://github.com/huggingface/transformers/blob/198c335d219a5eb4d3f124fdd1ce1a9cd9f78a9b/src/transformers/models/roberta/modeling_roberta.py#L1128"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.dense = nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.model_config.hidden_size, eps=float(self.model_config.language_encoder.layer_norm_eps))

        self.gelu = nn.GELU()
        self.decoder = nn.Linear(self.model_config.hidden_size, self.model_config.language_encoder.vocab_size)
        self.bias = nn.Parameter(torch.zeros(self.model_config.language_encoder.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias
