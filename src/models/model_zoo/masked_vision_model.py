"""
This script contains the code for 2 Models:
Masked Vision Model, based on the following: https://arxiv.org/abs/2111.06377

And the base ViT model used for downstream classification.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from pathlib import Path
import random
import timm
import timm.optim.optim_factory as optim_factory
from functools import partial
import os 
from collections import OrderedDict
from omegaconf.errors import ConfigAttributeError

from src.models.modules.pos_embeds import *
from src.models.base_model import BaseModel
from src.models.modules.image_encoder import *
from src.models.modules.masked_vision_layers import *
from src.models.modules.discriminators import Discriminator, MSGDiscriminator
from src.models.modules.stylegan_layers import *
from src.common.registry import registry
from src.models.modules.layer_utils import *
from src.losses.image_reconstruction import MaskedImageLoss, scale_pyramid
from src.datasets.transforms.vision_transforms_utils import UnNormalise
from src.common.constants import TRACTABLE_CED_MEAN, TRACTABLE_CED_STD, IMAGE_COLOR_MEAN, IMAGE_COLOR_STD
from torchmetrics import Accuracy, Precision, F1Score, Recall, AUROC

@registry.register_model("masked_image_autoencoder")
class MaskedImageAutoEncoder(BaseModel):
    """
    based on the paper: https://arxiv.org/abs/2111.06377
    Masked Autoencoder with VisionTransformer backbone

    From my experiments, model works better when image inputs are NOT Normalised!
    """
    def __init__(self, config, local_experiment_data_dir):
        super().__init__()
        print('for this work well, make sure inputs are not normalised!')
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config =  self.config.dataset_config
        self.user_config = self.config.user_config
        self.image_out_dir = os.path.join(local_experiment_data_dir, 'mae_recon_out')
        if not os.path.exists(self.image_out_dir):
            os.makedirs(self.image_out_dir)
        # patch embed args;
        self.mask_ratio = self.model_config.mask_ratio
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
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = MAEEncoder(config, self.patch_embed, self.norm_layer)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder = MAEDecoder(config, self.patch_embed, self.norm_layer)
        # --------------------------------------------------------------------------
        # Downstream training specifics
        #self.linear_probe= nn.Linear()

        # --------- build loss ---------
        self.loss_fnc = MaskedImageLoss(config, self.patch_embed)
        # if using the GAN loss; initiate the discriminator;
        if self.model_config.loss_type=='gan':
            raise Exception('To use the GAN loss, use the following model: {}. This model implementation does not support GAN loss'.format(
                            'masked_image_autoencoder_gan_loss'
            ))
        
        if self.model_config.normalisation_params=='imagenet':
            self.unnormalise = UnNormalise(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD)
        else:
            self.unnormalise = UnNormalise(TRACTABLE_CED_MEAN, TRACTABLE_CED_STD)

        if self.finetune_imagenet!=None:
            self.load_imagenet_weights()
            print('og imagenet weights loaded from: {} \n to commence finetuning'.format(self.finetune_imagenet))

    def load_imagenet_weights(self):
        # load the model dictionary
        # NOTE: ONLY encoder weights should be loaded; decoder has to be trained from scratch for the specific data;
        # otherwise no point in doing MAE since imagenet distribution can also reconstruct different image types.
        #pretrained_dict= torch.load(self.finetune_imagenet)
        pretrained_dict= torch.load(self.finetune_imagenet)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        model_dict = self.encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.encoder.load_state_dict(model_dict)
        # if patch embed value != pretrained dict; throw an error; to ensure patchembed is correct.
        
    def forward(self, x):
        predictions, masks= [], []
        # loop over the images;
        for i in range(x.size(2)):
            imgs = x[:,:,i,:,:]
            # run encoder;
            latent, mask, ids_restore = self.encoder(imgs, self.mask_ratio)
            # run decoder;
            pred = self.decoder(latent, ids_restore)  # [N, L, p*p*3]
            # append outputs for each image;
            predictions.append(pred)
            masks.append(mask)
        
        if self.dataset_config.max_images>1:
            predictions = torch.stack(predictions).squeeze().permute(1,0,2,3)
            masks = torch.stack(masks).squeeze().permute(1,0,2)
        else:
            predictions = torch.stack(predictions).squeeze()
            masks = torch.stack(masks).squeeze()#.permute(1,0,2)
        
        return predictions, masks

    def training_step(self, batch, batch_idx):
        if self.dataset_config.dataset_name == 'imagenet':
            x, _ = batch[0], batch[1] # in imagenet sample[0]== image, sample[1]== class
        else:
            x, pad_mask, _ = batch['images'], batch['pad_mask'], batch['labels']
        
        # if running model on a normal image db i.e. imagenet / single inference
        if x.ndim == 4:
            # add a dim to make rest of forward compatible (saves from having multiple models)
            x = x.unsqueeze(2)

        out, mask = self(x)
        loss=0.0 # initiate loss variable as 0 then add to it in a loop'
        for i in range(x.size(2)):
            # imgs, pred, mask
            if self.dataset_config.max_images>1:
                loss += self.loss_fnc(x[:,:,i,:,:], out[:,i,:,:], mask[:,i,:])
            else:
                loss += self.loss_fnc(x[:,:,i,:,:], out, mask)
        
        # clone logits for metrics (don't want gradients to pass)
        self.log('train_loss', loss.item(), rank_zero_only=True, prog_bar=True, logger=True)
        
        # log images;
        if self.global_step % self.frequency_to_visualise ==0:
            preds = out.clone().detach()
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
                                                                                        mask[rand_batch_id,j,:].unsqueeze(0), 
                                                                                        x[rand_batch_id,:,j,:,:].unsqueeze(0))
                else:
                    # only one image;
                    orig_img, masked_img, recon, recon_with_visible= self.visualise_sample(preds[rand_batch_id,:,:].unsqueeze(0), 
                                                                                           mask[rand_batch_id,:].unsqueeze(0), 
                                                                                           x[rand_batch_id,:,j,:,:].unsqueeze(0))
                
                Images.append(orig_img.permute(2,0,1))
                Masked_Images.append(masked_img.permute(2,0,1))
                Recons.append(recon.permute(2,0,1))
                ReconsVisible.append(recon_with_visible.permute(2,0,1))
            
            Images = torch.stack(Images)
            Masked_Images = torch.stack(Masked_Images)
            Recons = torch.stack(Recons)
            ReconsVisible = torch.stack(ReconsVisible)
            
            #print(Images.size(), Masked_Images.size(), Recons.size(), ReconsVisible.size())
            grid = make_grid(
                torch.cat((Images, Masked_Images, 
                           Recons, ReconsVisible), dim=0))
            #grid = make_grid(
            #    torch.cat((self.unnormalise(Images), self.unnormalise(Masked_Images), 
            #               self.unnormalise(Recons), self.unnormalise(ReconsVisible)), dim=0))
            
            #self.logger.experiment.add_image('train_images', grid, batch_idx, self.global_step)
            
            save_image(grid, '{}/{}.png'.format(self.image_out_dir, self.global_step))
        
        return loss
    
    def _eval_step(self, batch, batch_idx) -> float:
        if self.dataset_config.dataset_name == 'imagenet':
            x, _ = batch[0], batch[1] # in imagenet sample[0]== image, sample[1]== class
        else:
            x, pad_mask, y = batch['images'], batch['pad_mask'], batch['labels']

        # if running model on a normal image db i.e. imagenet / single inference
        if x.ndim == 4:
            # add a dim to make rest of forward compatible (saves from having multiple models)
            x = x.unsqueeze(2)

        out, mask = self(x)
        loss = 0.0 # initiate loss variable as 0 then add to it in a loop'
        for i in range(x.size(2)):
            # imgs, pred, mask
            if self.dataset_config.max_images > 1:
                loss += self.loss_fnc(x[:,:,i,:,:], out[:,i,:,:], mask[:,i,:])
            else:
                loss += self.loss_fnc(x[:,:,i,:,:], out, mask)

        # calculate other metrics
        # preds = out>0.5
        # metrics_output = self.calculate_metrics(out, y.int().squeeze(1)) 

        return loss
        
        # clone logits for metrics (don't want gradients to pass)
        self.log('val_loss', loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        # calculate other metrics
        #preds = out>0.5
        #metrics_output = self.calculate_metrics(out, y.int().squeeze(1)) 

    def validation_step(self, batch, batch_idx):
        loss = self._eval_step(batch, batch_idx)
        
        # clone logits for metrics (don't want gradients to pass)
        self.log('val_loss', loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._eval_step(batch, batch_idx)
        
        # clone logits for metrics (don't want gradients to pass)
        self.log('test_loss', loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        return {"test_loss": loss}

    
    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        weight_decay=0.05
        blr= 1.5e-4
        min_lr = 0.
        warmup_epochs=20
        betas= (0.9, 0.95)

        # following timm: set wd as 0 for bias and norm layers
        #param_groups = optim_factory.add_weight_decay(self, weight_decay) # weight_decay;
        optimizer = torch.optim.Adam(self.parameters(), lr=blr, betas=betas)
        #print(optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9) #build_scheduler(optimizer, self.config)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    # --------------- helper functions ----------------
    #def on_train_epoch_start(self):
    #    if self.current_epoch==0:
    #        sample_input= torch.randn((8,3,10,224,224))
    #        self.logger.experiment.add_graph(MaskedImageAutoEncoder(self.config),sample_input)

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

# ----------------------------- GAN VERSION OF MAE ---------------------------
@registry.register_model("masked_image_autoencoder_gan")
class MaskedImageAutoEncoderGAN(BaseModel):
    """
    GAN version of the model above
    """
    def __init__(self, config, local_experiment_data_dir):
        super().__init__()
        print('for this work well, make sure inputs are not normalised!')
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config =  self.config.dataset_config
        assert self.model_config.loss_type=='gan' or self.model_config.loss_type=='gan_perceptual', "GAN Loss type must be a gan to use this model!"

        self.image_out_dir = os.path.join(local_experiment_data_dir, 'mae_recon_out')
        if not os.path.exists(self.image_out_dir):
            os.makedirs(self.image_out_dir)
        # patch embed args;
        self.mask_ratio = self.model_config.mask_ratio
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

        # Define the Generator;
        self.generator = nn.ModuleDict({
            'patch_embed': PatchEmbed(img_size, self.patch_size, in_channels, embed_dim),
            'encoder': MAEEncoder(config, self.patch_embed, self.norm_layer),
            'decoder': MAEDecoder(config, self.patch_embed, self.norm_layer)
        })
        
        # Define the Discriminator
        self.discriminator = Discriminator(self.config) 

        # --------- build loss ---------
        self.loss_fnc = MaskedImageLoss(config, self.patch_embed, discriminator_model= self.discriminator, 
                                        G_mapping=self.generator.encoder, G_synthesis=self.generator.decoder, 
                                        latent_encoder=self.generator.encoder, do_ada=True, 
                                        augment_pipe=self.train_dataloader)
        
        print('loss loaded with discriminator')

        if self.finetune_imagenet!=None:
            self.load_imagenet_weights()
            print('og imagenet weights loaded from: {} \n to commence finetuning'.format(self.finetune_imagenet))

    def load_imagenet_weights(self):
        # load the model dictionary
        # NOTE: ONLY encoder weights should be loaded; decoder has to be trained from scratch for the specific data;
        # otherwise no point in doing MAE since imagenet distribution can also reconstruct different image types.
        pretrained_dict= torch.load(self.finetune_imagenet)
        model_dict = self.generator.encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.generator.encoder.load_state_dict(model_dict)
        # if patch embed value != pretrained dict; throw an error; to ensure patchembed is correct.
        
    def forward(self, x):
        predictions, masks= [], []
        # loop over the images;
        for i in range(x.size(2)):
            imgs = x[:,:,i,:,:]
            # run encoder;
            latent, mask, ids_restore = self.generator.encoder(imgs, self.mask_ratio)
            # run decoder;
            pred = self.generator.decoder(latent, ids_restore)  # [N, L, p*p*3]
            # append outputs for each image;
            predictions.append(pred)
            masks.append(mask)
        
        if self.dataset_config.max_images>1:
            predictions = torch.stack(predictions).squeeze().permute(1,0,2,3)
            masks = torch.stack(masks).squeeze().permute(1,0,2)
        else:
            predictions = torch.stack(predictions).squeeze()
            masks = torch.stack(masks).squeeze()#.permute(1,0,2)
        
        return predictions, masks

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.dataset_config.dataset_name == 'imagenet':
            x, _ = batch[0], batch[1] # in imagenet sample[0]== image, sample[1]== class
        else:
            x, pad_mask, _ = batch['images'], batch['pad_mask'], batch['labels']

        # if running model on a normal image db i.e. imagenet / single inference
        if x.ndim == 4:
            # add a dim to make rest of forward compatible (saves from having multiple models)
            x = x.unsqueeze(2)

        out, mask = self(x)
        disc_loss, gen_loss=0.0, 0.0 # initiate loss variable as 0 then add to it in a loop'
        
        for i in range(x.size(2)):
            # imgs, pred, mask
            if self.dataset_config.max_images>1:
                loss = self.loss_fnc(x[:,:,i,:,:], out[:,i,:,:], mask[:,i,:], self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]
            else:
                loss = self.loss_fnc(x[:,:,i,:,:], out, mask, self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]

            if len(loss) > 2:
                # log loss subcomponents
                for loss_name, loss_val in loss[2].items():
                    self.log(f'gen_loss_{loss_name}', loss_val, rank_zero_only=True, prog_bar=True)
        
        # perform back prop;
        # ------------ GENERATOR --------------
        if optimizer_idx==0:

            # clone logits for metrics (don't want gradients to pass)
            self.log('gen_loss', gen_loss, rank_zero_only=True, prog_bar=True)
            
            # log images;
            if self.global_step % self.frequency_to_visualise ==0:
                preds = out.clone().detach()
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
                                                                                            mask[rand_batch_id,j,:].unsqueeze(0), 
                                                                                            x[rand_batch_id,:,j,:,:].unsqueeze(0))
                    else:
                        # only one image;
                        orig_img, masked_img, recon, recon_with_visible= self.visualise_sample(preds[rand_batch_id,:,:].unsqueeze(0), 
                                                                                            mask[rand_batch_id,:].unsqueeze(0), 
                                                                                            x[rand_batch_id,:,j,:,:].unsqueeze(0))
                    
                    Images.append(orig_img.permute(2,0,1))
                    Masked_Images.append(masked_img.permute(2,0,1))
                    Recons.append(recon.permute(2,0,1))
                    ReconsVisible.append(recon_with_visible.permute(2,0,1))
                
                Images = torch.stack(Images)
                Masked_Images = torch.stack(Masked_Images)
                Recons = torch.stack(Recons)
                ReconsVisible = torch.stack(ReconsVisible)
                
                #print(Images.size(), Masked_Images.size(), Recons.size(), ReconsVisible.size())
                grid = make_grid(
                    torch.cat((Images, Masked_Images, 
                            Recons, ReconsVisible), dim=0))
                #grid = make_grid(
                #    torch.cat((self.unnormalise(Images), self.unnormalise(Masked_Images), 
                #               self.unnormalise(Recons), self.unnormalise(ReconsVisible)), dim=0))
                
                #self.logger.experiment.add_image('train_images', grid, batch_idx, self.global_step)
                
                save_image(grid, '{}/{}.png'.format(self.image_out_dir, self.global_step))
            
            return gen_loss
        
        # ------------ DISC Loss --------------
        if optimizer_idx==1:
            # clone logits for metrics (don't want gradients to pass)
            self.log('disc_loss', disc_loss, rank_zero_only=True, prog_bar=True)
            return disc_loss
            

    def validation_step(self, batch, batch_idx):
        if self.dataset_config.dataset_name == 'imagenet':
            x, _ = batch[0], batch[1] # in imagenet sample[0]== image, sample[1]== class
        else:
            x, pad_mask, y = batch['images'], batch['pad_mask'], batch['labels']

        # if running model on a normal image db i.e. imagenet / single inference
        if x.ndim == 4:
            # add a dim to make rest of forward compatible (saves from having multiple models)
            x = x.unsqueeze(2)

        out, mask = self(x)
        disc_loss, gen_loss=0.0, 0.0 # initiate loss variable as 0 then add to it in a loop'
        for i in range(x.size(2)):
            # imgs, pred, mask
            if self.dataset_config.max_images>1:
                loss = self.loss_fnc(x[:,:,i,:,:], out[:,i,:,:], mask[:,i,:], self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]
            else:
                loss = self.loss_fnc(x[:,:,i,:,:], out, mask, self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]
        

        # clone logits for metrics (don't want gradients to pass)
        self.log('val_loss', gen_loss, rank_zero_only=True, prog_bar=True)
        self.log('val_disc_loss', disc_loss, rank_zero_only=True, prog_bar=True)
        # calculate other metrics
        #preds = out>0.5
        #metrics_output = self.calculate_metrics(out, y.int().squeeze(1)) 

        return {'val_loss': gen_loss}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    
    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        weight_decay=0.05
        blr= 1.5e-4
        min_lr = 0.
        warmup_epochs=20
        betas= (0.9, 0.95)

        # following timm: set wd as 0 for bias and norm layers
        #param_groups = optim_factory.add_weight_decay(self, weight_decay) # weight_decay;
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=blr, betas=betas)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=blr, betas=betas)
        #print(optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=30, gamma=0.9) #build_scheduler(optimizer, self.config)
        return [optimizer_gen, optimizer_disc], [scheduler]

    # --------------- helper functions ----------------
    #def on_train_epoch_start(self):
    #    if self.current_epoch==0:
    #        sample_input= torch.randn((8,3,10,224,224))
    #        self.logger.experiment.add_graph(MaskedImageAutoEncoder(self.config),sample_input)

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

@registry.register_model("masked_image_autoencoder_msg_gan")
class MaskedImageAutoEncoderMSGGAN(BaseModel):
    """
    GAN version of the model above
    """
    def __init__(self, config, local_experiment_data_dir):
        super().__init__()
        print('for this work well, make sure inputs are not normalised!')
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config =  self.config.dataset_config
        assert self.model_config.loss_type=='gan' or self.model_config.loss_type=='gan_perceptual', "GAN Loss type must be a gan to use this model!"

        self.image_out_dir = os.path.join(local_experiment_data_dir, 'mae_recon_out')
        if not os.path.exists(self.image_out_dir):
            os.makedirs(self.image_out_dir)
        # patch embed args;
        self.mask_ratio = self.model_config.mask_ratio
        self.finetune_imagenet= self.model_config.finetune_imagenet
        self.num_samples_to_visualise = self.model_config.num_samples_to_visualise
        self.frequency_to_visualise = self.model_config.frequency_to_visualise
        self.scales = self.model_config.discriminator.depth

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

        # Define the Generator;
        self.generator = nn.ModuleDict({
            'patch_embed': PatchEmbed(img_size, self.patch_size, in_channels, embed_dim),
            'encoder': MSGMAEEncoder(config, self.patch_embed, self.norm_layer),
            'decoder': MSGMAEDecoder(config, self.patch_embed, self.norm_layer)
        })
        
        # Define the Discriminator
        self.discriminator = MSGDiscriminator(self.config) 

        # ------------------------------------ build loss ------------------------------------
        self.loss_fnc = MaskedImageLoss(config, self.patch_embed, discriminator_model= self.discriminator, 
                                        G_mapping=self.generator.encoder, G_synthesis=self.generator.decoder, latent_encoder=self.generator.encoder, 
                                        do_ada=True, augment_pipe=self.train_dataloader) # need to confirm; dataloader accessible from trainer
        
        print('loss loaded with discriminator')

        if self.finetune_imagenet!=None:
            self.load_imagenet_weights()
            print('og imagenet weights loaded from: {} \n to commence finetuning'.format(self.finetune_imagenet))

    def load_imagenet_weights(self):
        # load the model dictionary
        # NOTE: ONLY encoder weights should be loaded; decoder has to be trained from scratch for the specific data;
        # otherwise no point in doing MAE since imagenet distribution can also reconstruct different image types.
        pretrained_dict= torch.load(self.finetune_imagenet)
        model_dict = self.generator.encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.generator.encoder.load_state_dict(model_dict)
        # if patch embed value != pretrained dict; throw an error; to ensure patchembed is correct.
        
    def forward(self, x):
        predictions, masks, scaled_images= [], [], []
        # loop over the images;
        for i in range(x.size(2)):
            imgs = x[:,:,i,:,:]
            # run encoder; x, mask, ids_restore, prev_feat_maps
            latent, mask, ids_restore, prev_feat_maps = self.generator.encoder(imgs, self.mask_ratio)
            # run decoder;
            pred, rgb_outputs = self.generator.decoder(latent, ids_restore, prev_feat_maps)  # [N, L, p*p*3]
            # append outputs for each image;
            predictions.append(pred)
            masks.append(mask)
            scaled_images.append(rgb_outputs)
        
        if self.dataset_config.max_images>1:
            predictions = torch.stack(predictions).squeeze().permute(1,0,2,3)
            masks = torch.stack(masks).squeeze().permute(1,0,2)
        else:
            predictions = torch.stack(predictions).squeeze()
            masks = torch.stack(masks).squeeze()#.permute(1,0,2)
        
        return predictions, masks, scaled_images

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.dataset_config.dataset_name == 'imagenet':
            x, _ = batch[0], batch[1] # in imagenet sample[0]== image, sample[1]== class
        else:
            x, pad_mask, _ = batch['images'], batch['pad_mask'], batch['labels']

        # if running model on a normal image db i.e. imagenet / single inference
        if x.ndim == 4:
            # add a dim to make rest of forward compatible (saves from having multiple models)
            x = x.unsqueeze(2)

        out, mask, scaled_images = self(x)
    
        disc_loss, gen_loss=0.0, 0.0 # initiate loss variable as 0 then add to it in a loop'
        for i in range(x.size(2)):
            # imgs, pred, mask
            if self.dataset_config.max_images>1:
                scaled_gt = scale_pyramid(x[:,:,i,:,:], self.scales)
                
                loss = self.loss_fnc(scaled_gt[::-1], scaled_images[i][::-1], mask[:,i,:], self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]
            else:
                scaled_gt = scale_pyramid(x[:,:,i,:,:], self.scales)
                loss = self.loss_fnc(scaled_gt[::-1], scaled_images[i][::-1], mask, self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]

        if len(loss) > 2:
            # log loss subcomponents
            for loss_name, loss_val in loss[2].items():
                self.log(f'gen_loss_{loss_name}', loss_val, rank_zero_only=True, prog_bar=True)

        # perform back prop;
        # ------------ GENERATOR --------------
        if optimizer_idx==0:

            # clone logits for metrics (don't want gradients to pass)
            self.log('gen_loss', gen_loss.item(), rank_zero_only=True, prog_bar=True, logger=True)
            
            # log images;
            if self.global_step % self.frequency_to_visualise ==0:
                preds = out.clone().detach()
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
                                                                                            mask[rand_batch_id,j,:].unsqueeze(0), 
                                                                                            x[rand_batch_id,:,j,:,:].unsqueeze(0))
                    else:
                        # only one image;
                        orig_img, masked_img, recon, recon_with_visible= self.visualise_sample(preds[rand_batch_id,:,:].unsqueeze(0), 
                                                                                            mask[rand_batch_id,:].unsqueeze(0), 
                                                                                            x[rand_batch_id,:,j,:,:].unsqueeze(0))
                    
                    Images.append(orig_img.permute(2,0,1))
                    Masked_Images.append(masked_img.permute(2,0,1))
                    Recons.append(recon.permute(2,0,1))
                    ReconsVisible.append(recon_with_visible.permute(2,0,1))
                
                Images = torch.stack(Images)
                Masked_Images = torch.stack(Masked_Images)
                Recons = torch.stack(Recons)
                ReconsVisible = torch.stack(ReconsVisible)
                
                #print(Images.size(), Masked_Images.size(), Recons.size(), ReconsVisible.size())
                grid = make_grid(
                    torch.cat((Images, Masked_Images, 
                            Recons, ReconsVisible), dim=0))
                #grid = make_grid(
                #    torch.cat((self.unnormalise(Images), self.unnormalise(Masked_Images), 
                #               self.unnormalise(Recons), self.unnormalise(ReconsVisible)), dim=0))
                
                #self.logger.experiment.add_image('train_images', grid, batch_idx, self.global_step)
                
                save_image(grid, '{}/{}.png'.format(self.image_out_dir, self.global_step))
            
            return gen_loss
        
        # ------------ DISC Loss --------------
        if optimizer_idx==1:
            # clone logits for metrics (don't want gradients to pass)
            self.log('disc_loss', disc_loss.item(), rank_zero_only=True, prog_bar=True, logger=True)
            return disc_loss
            

    def validation_step(self, batch, batch_idx):
        if self.dataset_config.dataset_name == 'imagenet':
            x, _ = batch[0], batch[1] # in imagenet sample[0]== image, sample[1]== class
        else:
            x, pad_mask, _ = batch['images'], batch['pad_mask'], batch['labels']

        # if running model on a normal image db i.e. imagenet / single inference
        if x.ndim == 4:
            # add a dim to make rest of forward compatible (saves from having multiple models)
            x = x.unsqueeze(2)

        out, mask, scaled_images = self(x)
    
        disc_loss, gen_loss=0.0, 0.0 # initiate loss variable as 0 then add to it in a loop'
        
        for i in range(x.size(2)):
            # imgs, pred, mask
            if self.dataset_config.max_images>1:
                scaled_gt = scale_pyramid(x[:,:,i,:,:], self.scales)
                
                loss = self.loss_fnc(scaled_gt[::-1], scaled_images[i][::-1], mask[:,i,:], self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]
            else:
                scaled_gt = scale_pyramid(x[:,:,i,:,:], self.scales)
                loss = self.loss_fnc(scaled_gt[::-1], scaled_images[i][::-1], mask, self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]
        

        # clone logits for metrics (don't want gradients to pass)
        self.log('val_loss', gen_loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_disc_loss', disc_loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        # calculate other metrics
        #preds = out>0.5
        #metrics_output = self.calculate_metrics(out, y.int().squeeze(1)) 

        return {'val_loss': gen_loss}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    
    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        weight_decay=0.05
        blr= 1.5e-4
        min_lr = 0.
        warmup_epochs=20
        betas= (0.9, 0.95)

        # following timm: set wd as 0 for bias and norm layers
        #param_groups = optim_factory.add_weight_decay(self, weight_decay) # weight_decay;
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=blr, betas=betas)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=blr, betas=betas)
        #print(optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=30, gamma=0.9) #build_scheduler(optimizer, self.config)
        return [optimizer_gen, optimizer_disc], [scheduler]

    # --------------- helper functions ----------------
    #def on_train_epoch_start(self):
    #    if self.current_epoch==0:
    #        sample_input= torch.randn((8,3,10,224,224))
    #        self.logger.experiment.add_graph(MaskedImageAutoEncoder(self.config),sample_input)

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

# ----------------------------- ViT Model for Downstream FineTuning ------------------------------
@registry.register_model("vit_downstream")
class VITEncoderDownStream(BaseModel):
    def __init__(self, config, local_experiment_data_dir):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.data_type = self.model_config.data_type
        self.dataset_config =  self.config.dataset_config
        self.user_config = self.config.user_config
        self.transformer_params= self.model_config.transformer
        self.train_task = self.model_config.train_task
        self.gpu_device = self.config.trainer.params.gpus
        if self.gpu_device==-1:
            self.device_count = torch.cuda.device_count()
        else:
            self.device_count = len(self.gpu_device)

        try:
            loss_type_str = f'_{self.model_config.loss_type}'
        except ConfigAttributeError:
            loss_type_str = ''

        self.output_dir = os.path.join(local_experiment_data_dir, 'mae_downstream_out')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # encoder args;
        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        num_heads = self.model_config.image_encoder.num_heads
        mlp_ratio = self.model_config.image_encoder.mlp_ratio
        depth = self.model_config.image_encoder.depth

        self.norm_layer_arg= self.model_config.norm_layer_arg
        
        if self.norm_layer_arg=='partial':
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
            print('using partial layer norm')
        else:
            self.norm_layer = nn.LayerNorm

        # if using public datasets i.e. ImageNet / CIFAR etc, then original VIT model can be used.
        if self.dataset_config.dataset_name == 'imagenet' or self.dataset_config.dataset_name == 'imagenet_vision':
            self.patch_embedx = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
            num_patches = self.patch_embedx.num_patches
            
            self.vit_model = VisionTransformer(patch_size=self.patch_size, embed_dim=embed_dim, 
                                            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                            qkv_bias=True,
                                            norm_layer=self.norm_layer)
            self.vit_model.patch_embed = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
            self.vit_model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            # change the final layer classifier; to desired classes!
            # self.vit_model.head = nn.Linear(in_features=768, out_features=self.model_config.classifier.num_classes, bias=True)

        # If using Tractable data classification model is different to account for multiple images;
        else:
            self.vit_model = VisionTransformer_EncoderOnly(self.config, self.norm_layer, self.transformer_params.dropout_rate)
            self.linearize = nn.Conv2d(self.vit_model.num_patches + 1, 1, 1) # add 1 to patch to account for posembed


            self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model = self.transformer_params.d_model, 
                                        nhead=self.transformer_params.nhead, 
                                        dim_feedforward=self.transformer_params.dim_feedforward, 
                                        dropout= self.transformer_params.dropout_rate,
                                        batch_first=True)
            for i in range(self.transformer_params.depth)])

            self.transformer_layers = nn.Sequential(*self.transformer_blocks)

            #self.transformer_ = nn.TransformerEncoderLayer(d_model = self.transformer_params.d_model, 
            #                                            nhead=self.transformer_params.nhead, 
            #                                            dim_feedforward=self.transformer_params.dim_feedforward, 
            #                                            dropout= self.transformer_params.dropout_rate,
            #                                            batch_first=True) #256, 8, 512,

            #self.transformer_2 = nn.TransformerEncoderLayer(d_model = self.transformer_params.d_model, 
            #                                            nhead=self.transformer_params.nhead, 
            #                                            dim_feedforward=self.transformer_params.dim_feedforward, 
            #                                            dropout= self.transformer_params.dropout_rate,
            #                                            batch_first=True) #256, 8, 512,
       
            # -------- final fully connected --------;
            self.attn_pooling = nn.Conv2d(self.config.dataset_config.max_images, 1, 1)
            self.act = nn.ReLU()
            self.output_layer = nn.Linear(self.model_config.classifier.in_dim, self.model_config.classifier.num_classes)
        
        if self.model_config.load_pretrained_mae is not None:
            self.load_pre_text_pretrained_weights()
            print('pre-text MAE model weights loaded! \n from: {}'.format(self.model_config.load_pretrained_mae))
        else:
            print('no pre-text weights added; initiating with random weights')

        # if linear probing; freeze encoder, otherwise train as normal.
        if self.train_task=='linear_probe':
            self.vit_model = self.freeze_model(self.vit_model)
            print('LINEAR PROBE SELECTED! HENCE, FREEZING VIT ENCODER MODEL')
        else:
            print('FINE TUNING! HENCE, VIT ENCODER MODEL WILL BE TRAINED AS NORMAL')

        # -------- metrics ----------
         #self.metrics = #build_metrics(self.config)
        self.Precision = Precision(threshold=0.54, average='samples')
        self.Recall = Recall(threshold=0.54, average='samples')
        self.F1 = F1Score(threshold=0.54, average='samples', mdmc_average='samplewise')
        #self.Auroc= AUROC(average='micro')
        self.Accuracy = Accuracy(threshold=0.54, average='samples')
        # -------- loss function ----------
        if self.dataset_config.dataset_name == 'imagenet' or self.dataset_config.dataset_name == 'imagenet_vision':
            self.loss_fnc = nn.CrossEntropyLoss()
        else:
            self.loss_fnc = nn.BCEWithLogitsLoss()
        
        self.sigmoid = nn.Sigmoid()

    def load_pre_text_pretrained_weights(self):
        # load pretrained weights (from local);
        if self.model_config.load_pretrained_mae[0] == '/':
            pretrained_weights_path = self.model_config.load_pretrained_mae
        else:
            pretrained_weights_path = os.path.join(self.user_config.data_dir, self.model_config.load_pretrained_mae)
        pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')
        pretrained_weights = pretrained_weights['state_dict']
        # these weights will have encoder attached in front of the dict keys.
        # we will clean this up;
        new_pretrained_weights_dict = OrderedDict()
        for k, v in pretrained_weights.items():
            name = k.replace('encoder.', '') # remove `encoder.` k[8:]
            #name = 'vit_model.'+ name # add `vit_model.` to make it identical to current model.
            new_pretrained_weights_dict[name] = v
        # now that weights are identical between vit_model and pretrained encoder weights;
        model_dict = self.vit_model.state_dict()
        # 1. filter out unnecessary keys
        new_pretrained_weights_dict = {k: v for k, v in new_pretrained_weights_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_weights_dict) 
        # 3. load the new state dict
        self.vit_model.load_state_dict(model_dict)

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
        return model

    def forward(self, x, pad_mask=None):
        if self.dataset_config.dataset_name == 'imagenet' or self.dataset_config.dataset_name == 'imagenet_vision':
            logits = self.vit_model(x)
            return logits  #self.out_activation(logits)
        else:
            embeddings, embeddings_linearised = [], []
            
            # loop over the images;
            for i in range(x.size(2)):
                imgs = x[:,:,i,:,:]
                embed1 = self.vit_model(imgs)
                embeddings.append(embed1)
            embeddings = torch.stack(embeddings).permute(1,0,2,3)

            # linearize the patches:
            for j in range(embeddings.size(1)):
                embeddings_linearised.append(self.linearize(embeddings[:,j,:,:].unsqueeze(2)))
            embeddings_linearised = torch.stack(embeddings_linearised).squeeze()
            if embeddings_linearised.dim() < 3:
                embeddings_linearised = embeddings_linearised.unsqueeze(0)
            else:
                embeddings_linearised= embeddings_linearised.permute(1,0,2)

            # pass through transformer & classifier
            #embed2 = self.transformer_(embeddings_linearised, src_key_padding_mask= pad_mask.type(torch.cuda.FloatTensor))
            #embed2 = self.transformer_2(embed2, src_key_padding_mask= pad_mask.type(torch.cuda.FloatTensor))
            embed2 = self.transformer_layers(embeddings_linearised, src_key_padding_mask= pad_mask.type(torch.cuda.FloatTensor))
            embed2 = embed2.unsqueeze(2)
            embed2 = self.act(self.attn_pooling(embed2).squeeze(1))
            logits = self.output_layer(embed2.squeeze(1))
            
            return logits

    def training_step(self, batch, batch_idx):
        if self.dataset_config.dataset_name == 'imagenet':
            x, y = batch[0], batch[1]  # in imagenet sample[0]== image, sample[1]== class
            pad_mask = None
        else:
            x, pad_mask, y = batch['images'], batch['pad_mask'], batch['labels']
            y = y.squeeze(1)
        
        out = self.forward(x, pad_mask)
        loss = self.loss_fnc(out, y)

        #print(out.size(), y.size(), '{:.3f}'.format(loss.item()))
        # clone logits for metrics (don't want gradients to pass)
        self.log('train_loss', loss.item(), rank_zero_only=True, prog_bar=True, logger=True)
        
        # calculate metrics;
        preds = out.clone()

        f1_score = self.F1(preds, y.int())  
        precision_score = self.Precision(preds, y.int()) 
        recall_score = self.Recall(preds, y.int()) 
        accuracy_score = self.Accuracy(preds, y.int()) 
        #auroc_score = self.Auroc(preds, y.int().squeeze(1)) 

        self.log('train_{}'.format('f1'), f1_score, rank_zero_only=True)
        self.log('train_{}'.format('precision'), precision_score, rank_zero_only=True)
        self.log('train_{}'.format('recall'), recall_score, rank_zero_only=True)
        self.log('train_{}'.format('acurracy'), accuracy_score, rank_zero_only=True)
        #self.log('train_{}'.format('auroc'), auroc_score, rank_zero_only=True)

        #for metric_name, metric_val in metrics_output.items():
        #    self.log('train_{}'.format(metric_name), metric_val, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.dataset_config.dataset_name == 'imagenet' or self.dataset_config.dataset_name == 'imagenet_vision':
            x, y = batch
            pad_mask=None
        else:
            x, pad_mask, y = batch['images'], batch['pad_mask'], batch['labels']
            y = y.squeeze(1)

        out = self.forward(x, pad_mask)

        loss = self.loss_fnc(out, y)

        # calculate accuracy;
        self.Accuracy(out, y.int())
        # calculate other metrics
        #preds = out>0.5
        #metrics_output = self.calculate_metrics(out, y.int().squeeze(1)) 

        f1_score = self.F1(out, y.int())  
        precision_score = self.Precision(out, y.int()) 
        recall_score = self.Recall(out, y.int()) 
        #auroc_score = self.Auroc(out, y.int().squeeze(1)) 

        # clone logits for metrics (don't want gradients to pass)
        self.log('val_loss', loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('val_{}'.format('f1'), f1_score, rank_zero_only=True, logger=True, sync_dist=True)
        self.log('val_{}'.format('precision'), precision_score, rank_zero_only=True, logger=True, sync_dist=True)
        self.log('val_{}'.format('recall'), recall_score, rank_zero_only=True, logger=True, sync_dist=True)
        #self.log('val_{}'.format('auroc'), auroc_score, rank_zero_only=True)

        # log to logger;
        #for metric_name, metric_val in metrics_output.items():
        #    self.log('val_{}'.format(metric_name), metric_val, rank_zero_only=True)

        self.log("val_acc", self.Accuracy, prog_bar=True, rank_zero_only=True, logger=True, sync_dist=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        if self.dataset_config.dataset_name == 'imagenet' or self.dataset_config.dataset_name == 'imagenet_vision':
            x, y = batch
            pad_mask=None
        else:
            x, pad_mask, y = batch['images'], batch['pad_mask'], batch['labels']
            claim_id = data['claim_id']
            y = y.squeeze(1)

        # get the output types;
        logits = self.forward(x, pad_mask)
        output = self.sigmoid(logits)
        predictions = (output>0.55).float()

        bce_loss = self.loss_fnc(logits, y)
        # clone logits for metrics (don't want gradients to pass)
        self.log('test_loss', bce_loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)

        ##### Add your analytics code here, use self.log() to put it in the loading bar! (can be same as val / more)

        
        # Return dictionary (as many elements as you'd like;)
        return {'test_loss': bce_loss.item(), 'logits': logits.cpu().numpy(), 'preds': predictions.cpu().numpy(), 
                'outputs': output.cpu().numpy(), 'gt_labels': y.cpu().numpy(), 'claim_id': claim_id}

    def test_epoch_end(self, outputs):
        # this is a list of dictionaries
        logits_list = [d['logits'] for d in outputs]
        preds_list = [d['preds'] for d in outputs]
        outputs_list = [d['outputs'] for d in outputs]
        gt_labels_list = [d['gt_labels'] for d in outputs]
        claim_id_list = [d['claim_id'] for d in outputs]

        print('saving numpy arrays as backup')
        np.save('{}/pred_outputs.npy'.format(self.output_dir), np.asarray(preds_list))
        np.save('{}/gt_labels.npy'.format(self.output_dir), np.asarray(gt_labels_list))
        np.save('{}/logits.npy'.format(self.output_dir), np.asarray(logits_list))
        np.save('{}/sigmoid_outputs.npy'.format(self.output_dir), np.asarray(outputs_list))
        np.save('{}/claim_ids.npy'.format(self.output_dir), np.asarray(claim_id_list))

    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        betas= (0.9, 0.95)
        base_batch = 2
        lr = ((0.001 * self.device_count) * base_batch) / 8
        min_lr = 0.0001
        steps = self.trainer.total_training_steps
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=betas)
        optimizer = torch.optim.SGD(self.parameters(), lr = lr, momentum=0.99, nesterov=True)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9) #build_scheduler(optimizer, self.config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=min_lr)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}