from src.models.modules.layer_utils import * 
import torch 
import torch.nn as nn 
from src.models.modules.bert_model import BertCrossLayer, BertAttention
from src.models.modules.output_heads import *
from src.models.modules.layer_utils import get_attn_config

class FusionLayer(nn.Module):
    def __init__(self, config):
        super(FusionLayer, self).__init__()
        self.config = config
        self.model_config = self.config.model_config

        if self.model_config.fusion.type=='cross_attention':
            self.layer = CrossAttention(self.config)
        else:
            raise Exception("Merged Attention is yet to be implmented!")
            #self.layer = MergedAttention(self.config)
    
    def forward(self, image_embeds, image_masks, extend_image_masks, text_embeds, text_masks, extend_text_masks):
        return self.layer(image_embeds, image_masks, extend_image_masks, text_embeds, text_masks, extend_text_masks)


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.config = config
        self.model_config = self.config.model_config

        self.token_type_embeddings = nn.Embedding(2, self.model_config["hidden_size"])
        self.token_type_embeddings.apply(self.init_weights)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.extend_image_embeds = nn.Conv2d(50,196,1)
        self.compress_image_embeds = nn.Conv2d(196,50,1)
        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(get_attn_config(self.config)) for _ in range(self.model_config.fusion['num_top_layer'])])
        self.cross_modal_image_layers.apply(self.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(get_attn_config(self.config)) for _ in range(self.model_config.fusion['num_top_layer'])])
        self.cross_modal_text_layers.apply(self.init_weights)

        self.cross_modal_image_pooler = Pooler(self.model_config["hidden_size"])
        self.cross_modal_image_pooler.apply(self.init_weights)
        self.cross_modal_text_pooler = Pooler(self.model_config["hidden_size"])
        self.cross_modal_text_pooler.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, image_embeds, image_masks, extend_image_masks, text_embeds, text_masks, extend_text_masks):
        image_embeds_extend = self.extend_image_embeds(image_embeds.permute(0,2,1,3))
        image_embeds_extend = image_embeds_extend.permute(0,2,1,3)

        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))
        all_image_embeds=[]
        for embed_id in range(image_embeds.size(1)): #torch.Size([16, 5, 50, 768])
            all_image_embeds.append(image_embeds_extend[:,embed_id,:,:].unsqueeze(1) + self.token_type_embeddings(torch.full_like(input=image_masks[:,embed_id,:].unsqueeze(1), 
                                                                                                                           fill_value=1,
                                                                                                                           dtype=torch.int64)))
        
        image_embeds_extend = torch.stack(all_image_embeds)
        if image_embeds_extend.size(2)==1:
            image_embeds_extend = image_embeds_extend.squeeze(2).permute(1,0,2,3)
        else:
            image_embeds_extend = image_embeds_extend.squeeze().permute(1,0,2,3)
        
        # now that they are re-embedded; for co-attention. Put them through.
        x, y_all = text_embeds, image_embeds_extend
        #new_y_all=[]
        for img_id in range(image_embeds_extend.size(1)):
            y = y_all[:,img_id,:,:]
            for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
                x1 = text_layer(x, y, extend_text_masks, extend_image_masks[:,img_id,:,:,:])
                y1 = image_layer(y, x, extend_image_masks[:,img_id,:,:,:], extend_text_masks)
                x, y = x1[0], y1[0]
            #new_y_all.append(y)
            y_all[:,img_id,:,:]= y
        
        # re define the embeds, to be the new merged embeds;
        text_feats, image_feats = x, y_all

        cls_feats_text = self.cross_modal_text_pooler(x)
        cls_feats_multi_image= []
        # need to get new cls_image_feats for each image;;
        for img_idx in range(image_feats.size(1)):
            image_feats_idx = image_feats[:,img_idx,:,:]
            avg_image_feats = self.avgpool(image_feats_idx.transpose(1, 2)).view(image_feats_idx.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
            # used for itm;
            cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
            cls_feats_multi_image.append(cls_feats)
        
        # used for itm;
        cls_feats_multi_image = torch.stack(cls_feats_multi_image).permute(1,0,2)

        # recompress image to original; so that decoder can reconstruct.
        image_feats = self.compress_image_embeds(image_feats.permute(0,2,1,3))
        image_feats = image_feats.permute(0,2,1,3)

        output = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_feats_multi_image":cls_feats_multi_image, 
            "text_masks": text_masks,
            "image_masks": image_masks,
        }

        return output


"""
class MergedAttention(nn.Module):
    def __init__(self, config):
        super(MergedAttention, self).__init__()
        self.config = config
        self.model_config = self.config.model_config
        

        self.token_type_embeddings = nn.Embedding(2, self.model_config["hidden_size"])
        self.token_type_embeddings.apply(self.init_weights)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fusion_transformer = build_image_encoder(self.model_config.fusion.transformer, pretrained=False)
        
        self.cross_modal_image_pooler = Pooler(self.model_config["hidden_size"])
        self.cross_modal_image_pooler.apply(self.init_weights)
        self.cross_modal_text_pooler = Pooler(self.model_config["hidden_size"])
        self.cross_modal_text_pooler.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, image_embeds, image_masks, extend_image_masks, text_embeds, text_masks, extend_text_masks):
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, 1)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.fusion_transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.fusion_transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )

        cls_feats = self.pooler(x)

        output = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "text_masks": text_masks,
        }
        
        return output
"""