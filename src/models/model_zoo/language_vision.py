import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from pathlib import Path
import random
from copy import deepcopy
import timm
import timm.optim.optim_factory as optim_factory
from functools import partial 
import os 
from collections import OrderedDict
from transformers.optimization import AdamW


from src.models.base_model import BaseModel
from src.models.modules.image_encoder import *
from src.models.modules.masked_vision_layers import *
from src.models.modules.language_encoder import *
from src.models.modules.decoder import *
from src.models.modules.fusion_layer import *
from src.models.modules.output_heads import *
from src.losses.image_reconstruction import MaskedImageLoss
from src.losses.language_vision_losses import MaskedLanguageVisionLoss, MLMLoss, ITMLoss
from src.utils.builder import build_optimizer, build_scheduler, build_loss
from src.common.registry import registry
from src.models.modules.layer_utils import *


"""
In the merged attention module, the
text and visual features are simply concatenated together,
then fed into a single transformer block


In the co-attention
module, on the other hand, the text and visual features
are fed into different transformer blocks independently, and
techniques such as cross-attention are used to enable crossmodal interaction

the merged attention module is more parameter-efficient


Also Encoder-only model has shown to outperform Encoder-Decoder architectures (due to lack of decoupling
feature maps)
"""


@registry.register_model("language_vision_mae")
class LanguageVisionMAEModel(BaseModel):
    """
    text --> encoder() --> MLP --> embed
    image --> encoder() --> MLP --> embed
    """
    def __init__(self, config, experiment_data_dir):
        super(LanguageVisionMAEModel, self).__init__()
        print('for this work well, make sure inputs are not normalised!')
        self.config = config
        self.model_config = self.config.model_config
        self.language_model_params = self.model_config.language_encoder
        self.dataset_config = self.config.dataset_config
        self.image_out_dir = os.path.join(experiment_data_dir, 'lv_recon_out')
        if not os.path.exists(self.image_out_dir):
            os.makedirs(self.image_out_dir)
        
        # patch embed args;
        self.mask_ratio = self.model_config.image_mask_ratio
        self.finetune_imagenet= self.model_config.finetune_imagenet
        self.num_samples_to_visualise = self.model_config.num_samples_to_visualise
        self.frequency_to_visualise = self.model_config.frequency_to_visualise

        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        self.norm_layer_arg= self.model_config.norm_layer_arg
        
        if self.norm_layer_arg=='partial':
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
            print('using partial layer norm')
        else:
            self.norm_layer = nn.LayerNorm
        
        self.patch_embed = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)

        # ------ initiate Image model components ------
        self.image_encoder = MAEEncoder(self.config, self.patch_embed, self.norm_layer) # vit_model
        self.image_decoder= MAEDecoder(self.config, self.patch_embed, self.norm_layer) # MAE decoder head;
        
        if self.finetune_imagenet!=None:
            self.load_imagenet_weights()
            print('og imagenet weights loaded from: {} \n to commence finetuning'.format(self.finetune_imagenet))

        # ------ initiate Language model components ------
        #self.language_encoder = LanguageEncoder(self.config) # text_transformer
        self.language_encoder, self.language_model_config= build_language_encoder(arch=self.language_model_params.name,
                                                                                  task=self.language_model_params.task,
                                                                                  pretrained=self.language_model_params.pretrained, 
                                                                                  lang_model_config= None)

        self.language_output_head = RobertaMLMHead(self.config) # MLM output layer;

        # ------ initiate Fusion model components ------
        self.fusion_head = FusionLayer(self.config)
        #self.fuse_intra_image_embeds= IntraImageEmbeddingFusion(self.config)
        
        # ------ ITM head ------
        self.itm_output_head = ITMHead(self.config)

        # ------ helper layers -------
        self.cross_modal_text_transform = nn.Linear(self.model_config.fusion['input_text_embed_size'], self.model_config['hidden_size'])
        self.cross_modal_text_transform.apply(self.init_weights)
        self.cross_modal_image_transform = nn.Linear(self.model_config.fusion['input_image_embed_size'], self.model_config['hidden_size'])
        self.cross_modal_image_transform.apply(self.init_weights)
        
        # ------ initiate losses ------
        self.mlm_loss_fnc = MLMLoss(self.config)
        self.mim_loss_fnc = MaskedImageLoss(self.config, self.patch_embed)
        self.itm_loss_fnc = ITMLoss(self.config)
        #self.loss_fnc = MaskedLanguageVisionLoss(self.config, self.patch_embed)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_imagenet_weights(self):
        # load the model dictionary
        # NOTE: ONLY encoder weights should be loaded; decoder has to be trained from scratch for the specific data;
        # otherwise no point in doing MAE since imagenet distribution can also reconstruct different image types.
        pretrained_dict= torch.load(self.finetune_imagenet)
        if 'state_dict' in pretrained_dict:
            pretrained_dict['model'] = pretrained_dict['state_dict']
            del dictionary['state_dict']
        #if 'model' in pretrained_dict:
        model_dict = self.image_encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.image_encoder.load_state_dict(model_dict)
        # if patch embed value != pretrained dict; throw an error; to ensure patchembed is correct.
        

    def run_language_encoder(self, data_input):
        input_ids = data_input['masked_inputs']
        text_masks= data_input['attention_mask']
        # B x S x - x embed_size ; torch.Size([32, 3, 1, 512])
        #text_embeds = self.language_encoder.encoder.embeddings(input_ids=input_ids, attention_mask=text_masks)
        text_embeds = self.language_encoder(input_ids=input_ids, attention_mask=text_masks).last_hidden_state
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.language_encoder.get_extended_attention_mask(text_masks, input_shape, device)
        
        for layer in self.language_encoder.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        return text_embeds, text_masks, extend_text_masks

    def image_encoder_fwd(self, x):
        latents, masks, ids_restores= [], [], []
        # loop over the images;
        for i in range(x.size(2)):
            imgs = x[:,:,i,:,:]
            # run encoder;
            latent, mask, ids_restore = self.image_encoder(imgs, self.mask_ratio)
            # append outputs for each image;
            latents.append(latent)
            masks.append(mask)
            ids_restores.append(ids_restore)
        
        if self.dataset_config.max_images>1:
            latents = torch.stack(latents)
            masks = torch.stack(masks)
            ids_restores = torch.stack(ids_restores)
            if latents.dim()>1:
                if latents.size(1)==1:
                    latents = latents.permute(1,0,2,3)
                    masks = masks.permute(1,0,2)
                    ids_restores = ids_restores.permute(1,0,2)
                else:
                    latents = latents.squeeze().permute(1,0,2,3)
                    masks = masks.squeeze().permute(1,0,2)
                    ids_restores = ids_restores.squeeze().permute(1,0,2)

        else:
            latents = torch.stack(latents).squeeze()
            masks = torch.stack(masks).squeeze()#.permute(1,0,2)
            ids_restores = torch.stack(ids_restores).squeeze()
        
        return latents, masks, ids_restores

    def run_image_encoder(self, data_input):
        image = data_input['images']
        image_embeds, image_masks, ids_restores = self.image_encoder_fwd(image)
        device = image_embeds.device
        image_embeds = self.cross_modal_image_transform(image_embeds)
        # get masks similar to text embeddings; (so that encoder learns how to align masked feature representations)
        extend_image_masks=[]
        for i in range(image_masks.size(1)):
            img_mask = image_masks[:,i,:]
            input_shape = img_mask.size()
            extend_image_masks.append(self.language_encoder.get_extended_attention_mask(img_mask, input_shape, device))
        extend_image_masks = torch.stack(extend_image_masks).permute(1,0,2,3,4)

        return image_embeds, image_masks, extend_image_masks, ids_restores

    def run_image_decoder(self, image_embeddings, ids_restores):
        predictions=[]
        # run decoder;
        for i in range(image_embeddings.size(1)):
            latent = image_embeddings[:,i,:,:]
            ids_restore = ids_restores[:,i,:]
            pred = self.image_decoder(latent, ids_restore)  # [N, L, p*p*3] # self.decoder(latent, ids_restore)  # [N, L, p*p*3]
            predictions.append(pred)
        
        if self.dataset_config.max_images>1:
            predictions = torch.stack(predictions)
            if predictions.size(1)==1:
                predictions = predictions.permute(1,0,2,3)
            else:
                predictions = predictions.squeeze().permute(1,0,2,3)
        else:
            predictions = torch.stack(predictions).squeeze()
        
        return predictions

    def forward(self, data_input):
        # run language encoder:
        text_embeds, text_masks, extend_text_masks = self.run_language_encoder(data_input)
        # run image encoder:
        image_embeds, image_masks, extend_image_masks, ids_restores = self.run_image_encoder(data_input)

        # fusion: image_embeds, image_masks, extend_image_masks, text_embeds, text_masks, extend_text_masks
        fused_embeds = self.fusion_head(image_embeds, image_masks, extend_image_masks, text_embeds, text_masks, extend_text_masks)
        
        return fused_embeds, image_masks, ids_restores, text_masks

    def training_step(self, batch, batch_idx):
        # --------------------------- RUN Masked Modelling ---------------------------
        # gives you the fused new embeddings.
        fused_outputs, image_masks, ids_restores, text_masks = self(batch)
        # Now run them through the different decoders; to get your outputs
        
        # The Following are MLM Outputs;
        predicted_image = self.run_image_decoder(fused_outputs['image_feats'], ids_restores) 
        prediction_text_scores = self.language_output_head(fused_outputs['text_feats'])

        # --------------------------- RUN Image-Text Matching ---------------------------
        #itm_fused_outputs, itm_image_masks, itm_ids_restores, itm_text_masks = self(batch[1])
        # ITM Outputs;
        prediction_itm_score = self.itm_output_head(fused_outputs['cls_feats_multi_image'])

        # use the outputs to compute loss.
        # ------------- MIM Loss -----------------  
        mim_loss=0.0 # initiate loss variable as 0 then add to it in a loop'
        for i in range(batch['images'].size(2)):
            # imgs, pred, mask
            if self.dataset_config.max_images>1:
                mim_loss += self.mim_loss_fnc(batch['images'][:,:,i,:,:], predicted_image[:,i,:,:], image_masks[:,i,:])
            else:
                mim_loss += self.mim_loss_fnc(batch['images'][:,:,i,:,:], predicted_image, image_masks)
        
        # clone logits for metrics (don't want gradients to pass)
        self.log('train_mim_loss', mim_loss, rank_zero_only=True, prog_bar=True)

        # ------------- MLM Loss -----------------
        mlm_loss = self.mlm_loss_fnc(prediction_text_scores, batch['input_ids'])
        perplexity = torch.exp(mlm_loss)
        # clone logits for metrics (don't want gradients to pass)
        self.log('train_mlm_loss', mlm_loss, rank_zero_only=True, prog_bar=True)
        #self.log('train_accuracy', acc, rank_zero_only=True, prog_bar=True)
        self.log('train_mlm_perplexity', perplexity, rank_zero_only=True, prog_bar=True)
        
        # ------------- ITM Loss -----------------
        itm_loss = self.itm_loss_fnc(prediction_itm_score, batch['itm_labels'].float())
        self.log('train_itm_loss', itm_loss, rank_zero_only=True, prog_bar=True)

        # ------------- Final Loss -----------------
        final_loss = mim_loss + mlm_loss + itm_loss 
        self.log('train_final_loss', final_loss, rank_zero_only=True, prog_bar=True)
        
        # record image generation results
        if self.global_step % self.frequency_to_visualise ==0:
            preds = predicted_image.clone().detach()
            # randomly select a batch and n images
            if self.dataset_config.max_images>1:
                rand_img_ids = random.sample(range(0, self.dataset_config.max_images-1), self.num_samples_to_visualise)
                rand_batch_id= random.randint(0, self.config.training.batch_size-1)
            else:
                rand_img_ids= [0]
                rand_batch_id= 0
            Images, Masked_Images, Recons, ReconsVisible=[],[],[],[]
            
            
            for j in rand_img_ids:
                if self.dataset_config.max_images>1:
                    orig_img, masked_img, recon, recon_with_visible= self.visualise_sample(preds[rand_batch_id,j,:,:].unsqueeze(0), 
                                                                                        image_masks[rand_batch_id,j,:].unsqueeze(0), 
                                                                                        batch['images'][rand_batch_id,:,j,:,:].unsqueeze(0))
                else:
                    # only one image;
                    orig_img, masked_img, recon, recon_with_visible= self.visualise_sample(preds[rand_batch_id,:,:].unsqueeze(0), 
                                                                                           image_masks[rand_batch_id,:].unsqueeze(0), 
                                                                                           batch['images'][rand_batch_id,:,j,:,:].unsqueeze(0))
                
                Images.append(orig_img.permute(2,0,1))
                Masked_Images.append(masked_img.permute(2,0,1))
                Recons.append(recon.permute(2,0,1))
                ReconsVisible.append(recon_with_visible.permute(2,0,1))
            
            Images = torch.stack(Images)
            Masked_Images = torch.stack(Masked_Images)
            Recons = torch.stack(Recons)
            ReconsVisible = torch.stack(ReconsVisible)
            
            grid = make_grid(
                torch.cat((Images, Masked_Images, 
                           Recons, ReconsVisible), dim=0))
            save_image(grid, '{}/{}.png'.format(self.image_out_dir, self.global_step))

        return final_loss

    
    def validation_step(self, batch, batch_idx):
        # gives you the fused new embeddings.
        fused_outputs, image_masks, ids_restores, text_masks = self(batch)
        # Now run them through the different decoders; to get your outputs
        
        # The Following are MLM Outputs;
        predicted_image = self.run_image_decoder(fused_outputs['image_feats'], ids_restores) 
        prediction_text_scores = self.language_output_head(fused_outputs['text_feats'])

        # ITM Outputs;
        prediction_itm_score = self.itm_output_head(fused_outputs['cls_feats_multi_image'])

        # use the outputs to compute loss.
        # ------------- MIM Loss -----------------  
        mim_loss=0.0 # initiate loss variable as 0 then add to it in a loop'
        for i in range(batch['images'].size(2)):
            # imgs, pred, mask
            if self.dataset_config.max_images>1:
                mim_loss += self.mim_loss_fnc(batch['images'][:,:,i,:,:], predicted_image[:,i,:,:], image_masks[:,i,:])
            else:
                mim_loss += self.mim_loss_fnc(batch['images'][:,:,i,:,:], predicted_image, image_masks)
        
        # clone logits for metrics (don't want gradients to pass)
        self.log('val_mim_loss', mim_loss, rank_zero_only=True, prog_bar=True)

        # ------------- MLM Loss -----------------
        mlm_loss = self.mlm_loss_fnc(prediction_text_scores, batch['input_ids'])
        perplexity = torch.exp(mlm_loss)
        # clone logits for metrics (don't want gradients to pass)
        self.log('val_mlm_loss', mlm_loss, rank_zero_only=True, prog_bar=True)
        #self.log('train_accuracy', acc, rank_zero_only=True, prog_bar=True)
        self.log('val_mlm_perplexity', perplexity, rank_zero_only=True, prog_bar=True)
        # ------------- ITM Loss -----------------
        itm_loss = self.itm_loss_fnc(prediction_itm_score, batch['itm_labels'].float())
        self.log('val_itm_loss', itm_loss, rank_zero_only=True, prog_bar=True)
        
        # ------------- Final Loss -----------------
        final_loss = mim_loss + mlm_loss + itm_loss 
        self.log('val_loss', final_loss, rank_zero_only=True, prog_bar=True)

        return {'val_loss': final_loss}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    
    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        weight_decay=0.01
        decay_power = 1
        blr= 1.5e-5
        min_lr = 0.
        warmup_epochs=20
        betas= (0.9, 0.95)
        lr_mult_head = 5  # multiply lr for downstream heads
        lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

        # following timm: set wd as 0 for bias and norm layers
        #param_groups = optim_factory.add_weight_decay(self, weight_decay) # weight_decay;
        optimizer = torch.optim.Adam(self.parameters(), lr=blr, betas=betas)
        #print(optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9) #build_scheduler(optimizer, self.config)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_final_loss"}

    # --------------- helper functions ----------------
    #def on_train_epoch_start(self):
    #    if self.current_epoch==0:
    #        sample_input= torch.randn((8,3,10,224,224))
    #        self.logger.experiment.add_graph(MaskedImageAutoEncoder(self.config),sample_input)

    def copy_network(self, model, freeze=True):
        copy_net = copy.deepcopy(model)
        if freeze:
            for param in copy_net.parameters():
                param.requires_grad = False
        return copy_net
 
    
    def update_copied_weights(self, model_a, model_b):
        # function updates model_a's weights with model_b
        # make sure the model_a net has the same weights as the model_b network
        model_a.load_state_dict(self.model_b.state_dict())
        
    def visualise_sample(self, pred, mask, img):
        y = self.unpatchify(pred)
        y = torch.einsum('nchw->nhwc', y).detach() #.cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach() #.cpu()
        
        x = torch.einsum('nchw->nhwc', img)

        # masked image
        im_masked = x * (1 - mask)

        # model reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # return (original image, masked image, model reconstruction, fused; reconstruction + visible pixels)
        return x[0], im_masked[0], y[0], im_paste[0]

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

        # grid = torchvision.utils.make_grid(sample_imgs) 
        # self.logger.experiment.add_image('generated_images', grid, 0) 


# REDESIGNING THE DOWNSTREAM VERSION OF THE LANGUAGE VISION MODEL TO ACCOMODATE FOR CYCLE CONSISTENCY


class Perplexity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, output, target):
        ce = self.cross_entropy(output, target)
        perplexity = torch.exp(ce)
        return perplexity


