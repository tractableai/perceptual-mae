import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.registry import registry
import lpips
import copy
from torchvision import models
from timm.models.vision_transformer import PatchEmbed
from src.losses.dall_e.dvae import Dalle_VAE

@registry.register_loss('mae_loss')
class MaskedImageLoss(nn.Module):
    def __init__(self, config, patch_embed, discriminator_model=None, G_mapping=None, G_synthesis=None, latent_encoder=None, do_ada=None, augment_pipe=None):
        super(MaskedImageLoss, self).__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config = self.config.dataset_config
        self.image_loss_weightings=  self.model_config.image_loss_weightings
        self.gan_arch_type=  self.model_config.discriminator.gan_arch_type
        
        if discriminator_model != None:
            self.disc_model = discriminator_model
            self.disc_model_copy = copy.deepcopy(self.disc_model)
            self.disc_model_copy.load_state_dict(self.disc_model.state_dict())
            # Style GAN args;
            self.G_mapping=G_mapping
            self.G_synthesis=G_synthesis
            self.latent_encoder=latent_encoder
            self.do_ada=do_ada
            self.augment_pipe=augment_pipe
        
        self.scales = 4 
        # will be used in future for experimenting with different reconstruction losses
        self.loss_type = self.model_config.loss_type 
        
        # std (default) loss type is MSE based loss from the original paper.
        if self.loss_type=='ssim' or self.loss_type=='ms_ssim':
            self.ssim_loss = SSIM(self.config)
        
        elif self.loss_type=='perceptual':
            self.perceptual_loss = VanillaPerceptualLoss(self.config)

        elif self.loss_type=='gan' or self.loss_type=='gan_perceptual':
            self.gan_loss= GANLoss(self.config, self.disc_model, self.gan_arch_type,
                                   self.G_mapping, self.G_synthesis, self.latent_encoder,
                                   self.do_ada, self.augment_pipe)
        
        self.norm_pix_loss = self.model_config.norm_pix_loss
        
        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        

        self.patch_embed = patch_embed #PatchEmbed(img_size, patch_size, in_channels, embed_dim)
    
    def forward(self, imgs, pred, mask, epoch=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # OG Mean Square Error:
        if self.loss_type=='mae':
            target = self.patchify(imgs)

            # implements original https://arxiv.org/abs/2111.06377 loss
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5


            # original loss is L2
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            # change to L1
            # loss = (pred - target)
            # loss = loss.abs().mean(dim=-1)  # [N, L], mean loss per patch

            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        # SSIM + L1 Loss:
        elif self.loss_type=='ssim':
            # single scale structural similarity index loss with l1
            pred = self.unpatchify(pred)
            loss = self.ssim_loss(pred, imgs, self.scales)
        
        # L1 + Style + Perceptual Loss:
        elif self.loss_type=='perceptual':
            # has to be paired with l1 or equivalent! otherwise; results will be blocky.
            
            pred = self.unpatchify(pred)
            loss_dict= self.perceptual_loss(pred, imgs)
            # filter out dictionary with relevant keys;
            dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
            
            perc_dict_weightings = dictfilt(self.image_loss_weightings, loss_dict.keys())
            # calculate loss and multiply by weightings
            loss=0.0
            for key, coef in perc_dict_weightings.items():
                value = coef * loss_dict[key]
                loss += value
        
        # GAN Loss:
        elif self.loss_type=='gan':
            # pred, gt
            if self.gan_arch_type!='msg':
                pred = self.unpatchify(pred)
            else:
                pass
            loss = self.gan_loss(pred, imgs) # (disc_loss, gen_loss)
            # loss here returns both generator and discriminator loss as a tuple!
        
        # GAN + Perceptual Loss:
        elif self.loss_type=='gan_perceptual':
            # check current epoch; and update discriminator model:
            if epoch%5==0:
                self.disc_model_copy.load_state_dict(self.disc_model.state_dict())
                # now use this disc model weights to initialise perceptual loss
            # initialise new perceptual loss;
            self.perceptual_loss= GANPerceptualLoss(self.config, self.disc_model_copy, self.gan_arch_type)

            # Calculate GAN Loss with pred, gt
            if self.gan_arch_type!='msg':
                pred = self.unpatchify(pred)
            else:
                pass

            disc_loss, gen_loss = self.gan_loss(pred, imgs)
            perc_loss_dict = self.perceptual_loss(pred, imgs)

            perc_loss_dict_weightings = {k: v for k, v in self.image_loss_weightings.items() if k in perc_loss_dict}

            gen_loss_dict_weighted = {
                'gan': gen_loss
            }

            # sum the weighted perceptual loss
            perc_loss = 0.0
            for key, coef in perc_loss_dict_weightings.items():
                value = coef * perc_loss_dict[key]
                perc_loss += value
                gen_loss_dict_weighted[key] = value

            gen_loss_w_perc = gen_loss + perc_loss

            # print(f'gen_loss: {gen_loss}, perc_loss: {perc_loss} => {gen_loss_w_perc}')
            # print(f'gen_loss_dict_weighted: {gen_loss_dict_weighted}')
            # print(f'sum: {sum(gen_loss_dict_weighted.values())}')

            # add subcomponents as third item in tuple (for logging)
            loss = [disc_loss, gen_loss_w_perc, gen_loss_dict_weighted]

        return loss

    # ---------------- helper functions ------------------
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


@registry.register_loss('lpips')
class LPIPS(nn.Module):
    def __init__(self, config):
        super(LPIPS, self).__init__()
        """
        from: https://arxiv.org/abs/1801.03924
        """
        self.config = config
        self.loss_fn = lpips.LPIPS(net=self.config.model_config.loss_network) # best forward scores
        # i.e. vgg / alex etc.
    
    def forward(self, pred, gt):
        # both images should be RGB normalised to -1 to 1
        loss = self.loss_fn(pred, gt)
        return loss


@registry.register_loss('perceptual_and_style')
class VanillaPerceptualLoss(nn.Module):
    """
    from: https://arxiv.org/abs/1603.08155
    This loss gives you both Perceptual and Style Transfer loss
    as a dictionary output
    choice is yours whether to use both or just one.
    """
    def __init__(self, config):
        super(VanillaPerceptualLoss, self).__init__()
        self.config = config

        if self.config.model_config.feature_extractor=='dall_e':
            self.feat_extractor = DALLEFeatureExtractor(self.config)
            self.blocks=4
            print('using DALL-E encoder as feature extractor')
        else:
            self.feat_extractor = VGG16FeatureExtractor()
            self.blocks=3
            print('using VGG16 as feature extractor')

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        print('''FOR STYLE LOSS, TRAIN IN FULL PRECISION (FP32) NOT HALF PREDICION (FP16) \n
                 otherwise gram matrix calculation will result in inf values and loss will be nan''')

    def forward(self, pred, gt):
        losses={}

        losses['l1'] = self.l1(pred, gt)
        
        if pred.shape[1] == 3:
            feat_output = self.feat_extractor(pred)
            feat_gt = self.feat_extractor(gt)
        elif pred.shape[1] == 1:
            feat_output = self.feat_extractor(torch.cat([pred]*3, 1))
            feat_gt = self.feat_extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('input shape must be either a RGB image or GrayScale')

        # get perceptual loss;
        losses['perc'] = 0.0
        # We extracted feature maps from 3 scales of the VGG network;
        for i in range(self.blocks):
            losses['perc'] += self.l1(feat_output[i], feat_gt[i])

        losses['style'] = 0.0
        for i in range(self.blocks):
            losses['style'] += self.l1(gram_matrix(feat_output[i]),
                                       gram_matrix(feat_gt[i]))

        return losses


@registry.register_loss('gan_perceptual')
class GANPerceptualLoss(nn.Module):
    """
    from: https://arxiv.org/abs/1603.08155
    This loss gives you both Perceptual and Style Transfer loss
    as a dictionary output
    choice is yours whether to use both or just one.
    """
    def __init__(self, config, discriminator_model, arch_type):
        super(GANPerceptualLoss, self).__init__()
        self.config = config
        self.disc_model = discriminator_model
        self.arch_type = arch_type

        if self.arch_type=='msg':
            self.feat_extractor = MSGDiscFeatureExtractor(self.config, self.disc_model)
            self.blocks=self.config.model_config.discriminator.depth-1
            #print('using multi-scale discriminator as feature extractor')
        else:
            self.feat_extractor = DiscFeatureExtractor(self.config, self.disc_model)
            self.blocks=4
            #print('using single scale discriminator as feature extractor')

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        #print('''FOR STYLE LOSS, TRAIN IN FULL PRECISION (FP32) NOT HALF PREDICION (FP16) \n
        #         otherwise gram matrix calculation will result in inf values and loss will be nan''')

    def forward(self, pred, gt):
        losses={}
        if self.arch_type=='msg':
            losses['l1'] = self.l1(pred[-1], gt[-1])
            if pred[-1].shape[1] == 3:
                feat_output = self.feat_extractor(pred)
                feat_gt = self.feat_extractor(gt)
            elif pred[-1].shape[1] == 1:
                feat_output = self.feat_extractor(torch.cat([pred]*3, 1))
                feat_gt = self.feat_extractor(torch.cat([gt]*3, 1))
            else:
                raise ValueError('input shape must be either a RGB image or GrayScale')

        else:
            losses['l1'] = self.l1(pred, gt)
        
            if pred.shape[1] == 3:
                feat_output = self.feat_extractor(pred)
                feat_gt = self.feat_extractor(gt)
            elif pred.shape[1] == 1:
                feat_output = self.feat_extractor(torch.cat([pred]*3, 1))
                feat_gt = self.feat_extractor(torch.cat([gt]*3, 1))
            else:
                raise ValueError('input shape must be either a RGB image or GrayScale')

        # get perceptual loss;
        losses['perc'] = 0.0
        # We extracted feature maps from 3 scales of the VGG network;
        for i in range(self.blocks):
            losses['perc'] += self.l1(feat_output[i], feat_gt[i])

        losses['style'] = 0.0
        for i in range(self.blocks):
            losses['style'] += self.l1(gram_matrix(feat_output[i]),
                                       gram_matrix(feat_gt[i]))

        return losses
            
@registry.register_loss('psnr')
class PSNR(nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self, config):
        super(PSNR, self).__init__()
        self.config = config

    @staticmethod
    def __call__(pred, gt):
        mse = torch.mean((pred - gt) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

@registry.register_loss('ssim')
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, config):
        super(SSIM, self).__init__()
        self.config= config
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.ssim_w = 0.85

    def ssim(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


    def forward(self, pred, gt, scales):
        #get multi scales
        pred_pyramid = scale_pyramid(pred, scales)
        gt_pyramid = scale_pyramid(gt, scales)
        # calculate L1 loss:
        l1_loss = [torch.mean(torch.abs(pred_pyramid[i] - gt_pyramid[i])) for i in range(scales)]
        # calculate SSIM loss
        ssim_loss = [torch.mean(self.ssim(pred_pyramid[i], gt_pyramid[i])) for i in range(scales)]
        # combine SSIM and L1: (0.85 * SSIM Loss) + (0.15 * L1 Loss)
        image_loss = [self.ssim_w * ssim_loss[i] + (1 - self.ssim_w) * l1_loss[i] for i in range(scales)]
        # sum all loss tensors (For all the scales)
        image_loss = sum(image_loss)

        return image_loss

@registry.register_loss('gan')
class GANLoss(nn.Module):
    def __init__(self, config, discriminator, arch_type, G_mapping, G_synthesis, 
                 latent_encoder, do_ada, augment_pipe):
        super(GANLoss, self).__init__()
        assert type(discriminator)!=None, "need a discriminator network to perform adversarial training!"
        self.config = config
        self.model_config = self.config.model_config
        self.gen_w = self.model_config.image_loss_weightings.gan
        self.gan_loss_type = self.model_config.gan_loss_type
        # a binary classifier model
        self.disc = discriminator
        self.arch_type = arch_type # if msg gan or normal gan

        if self.gan_loss_type=='std' or self.gan_loss_type=='standard':
            # define the criterion object
            self.criterion = StandardGAN(self.disc)
        elif self.gan_loss_type=='ls' or self.gan_loss_type=='least_squares':
            # define the criterion object
            self.criterion = LSGAN(self.disc)
        
        elif self.gan_loss_type=='wgan' or self.gan_loss_type=='wasserstein':
            self.criterion = WGAN_GP(self.disc)

        elif self.gan_loss_type=='style' or self.gan_loss_type=='stylegan':
            self.criterion= StyleGAN2ADALoss(self.config, self.disc, G_mapping, G_synthesis, 
                                             latent_encoder, do_ada, augment_pipe)

        # add l1 loss to Generator loss;
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, gt):
        # discriminator loss;
        disc_loss = self.criterion.disc_loss(gt, pred)
        # generator loss;
        gen_loss_ = self.criterion.gen_loss(pred)
        if self.arch_type=='msg':
            l1_error = self.l1_loss(pred[-1], gt[-1])
        else:
            l1_error = self.l1_loss(pred, gt)
        # gen_loss + l1 loss;
        gen_loss = l1_error + (self.gen_w * gen_loss_)

        return [disc_loss, gen_loss]

# --------- Original GANs loss with Binary Cross Entropy Loss -----------
class StandardGAN(nn.Module):
    def __init__(self, discriminator):
        super(StandardGAN, self).__init__()

        # define the criterion and activation used for object
        self.disc = discriminator
        self.criterion = nn.BCEWithLogitsLoss()

    def disc_loss(self, real_samps, fake_samps):
        # make sure everything is on the same device;
        device = real_samps.device
        # predictions for real images and fake images separately :
        r_preds = self.disc(real_samps)
        f_preds = self.disc(fake_samps)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, fake_samps):
        # make sure everything is on the same device;
        device = fake_samps.device
        preds = self.disc(fake_samps)
        return self.criterion(torch.squeeze(preds),
                              torch.ones(fake_samps.shape[0]).to(device))

# --------- Wasserstein GANs loss with Gradient Penalty -----------
class WGAN_GP(nn.Module):
    def __init__(self, discriminator, drift=0.001, use_gp=False):
        super(WGAN_GP, self).__init__()
        self.drift = drift
        self.disc = discriminator
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        # make sure everything is on the same device;
        device = real_samps.device

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)
        merged.requires_grad = True

        # forward pass
        op = self.disc(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=torch.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps):
        # define the (Wasserstein) loss
        fake_out = self.disc(fake_samps)
        real_out = self.disc(real_samps)

        loss = (torch.mean(fake_out) - torch.mean(real_out)
                + (self.drift * torch.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps)
            loss += gp

        return loss

    def gen_loss(self, fake_samps):
        # calculate the WGAN loss for generator
        loss = -torch.mean(self.dis(fake_samps))

        return loss

# --------- Least Squares GANs loss -----------
class LSGAN(nn.Module):
    def __init__(self, discriminator):
        super(LSGAN, self).__init__()
        self.disc = discriminator

    def disc_loss(self, real_samps, fake_samps):
        return 0.5 * (((torch.mean(self.disc(real_samps)) - 1) ** 2)
                      + (torch.mean(self.disc(fake_samps))) ** 2)

    def gen_loss(self, fake_samps):
        return 0.5 * ((torch.mean(self.disc(fake_samps)) - 1) ** 2)
    
# --------- Style GAN v2 - ADA loss -----------
"""
Style Gan architecture essentially uses (1) Mapping network (f) to convert noise vector z to an intermediate latent space w
(2) a synthesis network (g) that generates the final image, given w. 
Style modulation operation is added to each layer of g to guide image generation.

So in the context of MaE, f = the encoder (converts patches to latent space w). g = the decoder (generates final image)
Essentially f learns to disentangle the latent space, allowing for more control for decoder to generate a high fidelity output.

The following reimplements the core aspect of StyleGAN2-ADA loss from: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/loss.py
However note, this paper re-purposes the loss for Masked Image Modelling Hence not all aspects of the original loss are implemented nor relevant. The key aspects
re-implemented are the following:
ADA, Lazy Regularization, Dynamic R1 Coefficient, Latent Reconstruction Loss, and Perceptual Path Length Regularization
i.e.
1. R1 Regularization: This penalizes the gradients of the discriminator's outputs with respect to the real samples. It helps stabilize training.
2. Lazy Regularization: Instead of computing the regularization at every step, it's computed at intervals to save computation.
3. Dynamic R1 Coefficient: Adjusts the R1 penalty coefficient based on training progress.
4. Path Length Regularization: Regularizes the generator to ensure that small changes in the latent space don't result in drastic changes in the output space.
5. Adaptive Data Augmentation (ADA): Augments data to improve training in limited data scenarios, with dynamic augmentation strength adjustment based on discriminator feedback.
6. Latent Reconstruction Loss: Aims to ensure that the encoder can map generated images back to their latent codes effectively.

All of these loss components and regularizations are captured within the class. However, remember that the effectiveness of these components also depends on hyperparameters, 
which might need careful tuning based on the dataset and desired outputs.
"""

class StyleGAN2ADALoss(nn.Module):
    def __init__(self, config, discriminator, G_mapping, G_synthesis, latent_encoder, do_ada, augment_pipe):
        super(StyleGAN2ADALoss, self).__init__()
        self.config = config
        self.style_gan_args = self.config.model_config.style_gan_args
        # Initialization
        self.disc = discriminator
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.latent_encoder = latent_encoder
        self.augment_pipe = augment_pipe
        self.criterion = nn.BCEWithLogitsLoss() # StandardGAN(self.disc)
        self.r1_gamma = self.style_gan_args.r1_gamma
        self.ADA = do_ada
        # Parameters for Perceptual Path Length regularisation
        self.pl_weight = self.style_gan_args.pl_weight
        self.pl_decay = self.style_gan_args.pl_decay
        self.pl_mean = torch.zeros([])  # Initialized on CPU for this environment
        self.pl_denorm = torch.tensor(1.0, device=torch.device('cuda'))
        # Parameters for ADA adjustment
        self.upper_threshold = self.style_gan_args.upper_threshold
        self.lower_threshold = self.style_gan_args.lower_threshold
        self.adjust_factor = self.style_gan_args.adjust_factor
        # Parameters for R1 regularisation
        self.r1_interval = self.style_gan_args.r1_interval
        self.r1_gamma_dynamic = self.style_gan_args.r1_gamma
        self.r1_gamma_initial = self.style_gan_args.r1_gamma
        self.recon_weight = self.style_gan_args.recon_weight
    
    def adjust_augmentation_strength(self, real_samps, fake_samps):
        """Adjust the strength of augmentations based on discriminator feedback.

            Parameters:
            - discriminator: The discriminator model.
            - real_samps: Real samples.
            - fake_samps: Fake samples generated by the generator.
            - augment_pipe: The augmentation pipeline with a tunable strength.
            - upper_threshold: The upper threshold for accuracy to increase augmentation strength.
            - lower_threshold: The lower threshold for accuracy to decrease augmentation strength.
            - adjust_factor: The amount by which to adjust the augmentation strength.
            
            Returns:
            - New augmentation strength.
        
        """
        device = real_samps.device
        
        # Get predictions
        r_preds = torch.sigmoid(self.disc(real_samps))
        f_preds = torch.sigmoid(self.disc(fake_samps))
        
        # Calculate accuracies
        real_accuracy = ((r_preds > 0.5).float().mean()).item()
        fake_accuracy = ((f_preds < 0.5).float().mean()).item()
        
        # Adjust augmentation strength
        if real_accuracy > self.upper_threshold or fake_accuracy > self.upper_threshold:
            self.augment_pipe.strength = min(self.augment_pipe.strength + self.adjust_factor, 1.0)
        elif real_accuracy < self.lower_threshold and fake_accuracy < self.lower_threshold:
            self.augment_pipe.strength = max(self.augment_pipe.strength - self.adjust_factor, 0.0)
    
        return self.augment_pipe.strength

    def path_length_regularization(self, fake_samps, latents):
        """Compute the path length regularization term."""
        import math
        # Compute |J*y|.
        pl_noise = torch.randn(fake_samps.shape, device=fake_samps.device) / math.sqrt(fake_samps.shape[2] * fake_samps.shape[3])
        pl_grads = torch.autograd.grad(outputs=fake_samps * pl_noise, inputs=latents,
                                       grad_outputs=torch.ones(fake_samps.shape, device=fake_samps.device),
                                       create_graph=True, retain_graph=True, only_inputs=True)[0]
        pl_lengths = torch.sqrt(pl_grads.pow(2).sum(2).mean(1))

        # Track exponential moving average of |J*y|.
        pl_mean_val = self.pl_mean + self.pl_denorm * (pl_lengths.mean() - self.pl_mean)
        self.pl_mean.mul_(1 - self.pl_denorm).add_(pl_mean_val * self.pl_denorm)
        self.pl_denorm.mul_(1 - self.pl_denorm)

        # Calculate (|J*y|-a)^2.
        pl_penalty = (pl_lengths - self.pl_mean).pow(2)
        return pl_penalty

    def latent_reconstruction_loss(self, real_samps):
        """Compute the latent reconstruction loss."""
        # Encode the real samples to the w space
        w_reconstructed = self.latent_encoder(real_samps)

        # Get the original w from the mapping network
        with torch.no_grad():
            z = torch.randn(real_samps.size(0), 512) # assuming z_dim is 512
            w_original = self.G_mapping(z)
        
        # Compute the reconstruction loss
        recon_loss = F.mse_loss(w_original, w_reconstructed)
        return recon_loss
    
    # STYLE MIXING IS NOT APPLICABLE IN THE CASE OF MAE.... HENCE NOT USED IN gen_loss()
    # FUNCTION PRESENTED HERE FOR REFERENCE / FUTURE WORK IF CAN BE UTILISED
    def style_mixing(self, z):
        # Draw two random batches of latent vectors
        z1, z2 = z, torch.randn_like(z)
        
        # Obtain the corresponding w vectors
        w1, w2 = self.G_mapping(z1), self.G_mapping(z2)
        
        # Choose a random layer to mix styles
        num_layers = len(self.G_synthesis)
        mix_layer = torch.randint(1, num_layers, [1]).item()
        
        # Mix the styles of w1 and w2
        w_mixed = [w1[i] if i < mix_layer else w2[i] for i in range(num_layers)]
        
        return w_mixed

    def disc_loss(self, real_samps, fake_samps, cur_step):
        """Compute the discriminator loss."""

        # Apply ADA augmentations
        if self.ADA:
            real_samps, _ = self.augment_pipe(real_samps)
            fake_samps, _ = self.augment_pipe(fake_samps)

        r_preds = self.disc(real_samps)
        f_preds = self.disc(fake_samps)

        # GAN Loss: Replace with GAN loss class if you want to try with other types of GAN Losses;
        real_loss = self.criterion(torch.squeeze(r_preds), torch.ones(real_samps.shape[0]))
        fake_loss = self.criterion(torch.squeeze(f_preds), torch.zeros(fake_samps.shape[0]))

        # Lazy Regularization and Dynamic R1 Coefficient
        r1_loss = 0
        if cur_step % self.r1_interval == 0:
            # R1 Regularization
            real_grads = torch.autograd.grad(outputs=r_preds.sum(), inputs=real_samps, create_graph=True)[0]
            r1_penalty = (real_grads.square().sum([1,2,3])).mean()
            r1_loss = self.r1_gamma_dynamic * r1_penalty

            # Adjust R1 gamma dynamically (example approach: increase by a small value)
            self.r1_gamma_dynamic += 0.01

        # Adjust the augmentation strength
        self.adjust_augmentation_strength(real_samps, fake_samps)
        # return (real_loss + fake_loss) / 2 is the GAN Loss term; change if using GAN Loss class
        return (real_loss + fake_loss) / 2 + r1_loss

    def gen_loss(self, fake_samps, encoder_embeddings):
        """Compute the generator loss without style mixing."""
        device = fake_samps.device
        """
        # if doing style mixing;
        # Decide whether to perform style mixing
        if torch.rand(1).item() < self.style_mix_prob:
            ws = self.style_mixing(z)
        else:
            ws = self.G_mapping(z, c)
        """
        #ws = self.G_mapping(z)  # Directly obtain ws without style mixing >>> used for perceptual path length regularization
        #fake_samps = self.G_synthesis(ws)
        preds = self.disc(fake_samps)
        # GAN Loss: Replace with GAN loss class if you want to try with other types of GAN Losses;
        gan_loss = self.criterion(torch.squeeze(preds), torch.ones(fake_samps.shape[0]).to(device))
        # Path Length Regularization
        pl_penalty = self.path_length_regularization(fake_samps, encoder_embeddings)
        # Latent Reconstruction Loss
        recon_loss = self.latent_reconstruction_loss(fake_samps)
        return gan_loss + self.pl_weight * pl_penalty + self.recon_weight * recon_loss


# ---------------------------- Helper Functions ----------------------------
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class DALLEFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(DALLEFeatureExtractor, self).__init__()
        self.config = config
        # loads the full dall_e model; encoder + decoder
        self.dall_e = Dalle_VAE(self.config)
        # select the encoder only for the feature extractor
        self.encoder = self.dall_e.encoder
        self.input_layer = self.encoder.blocks.input
        self.enc_1 = self.encoder.blocks.group_1
        self.enc_2 = self.encoder.blocks.group_2
        self.enc_3 = self.encoder.blocks.group_3
        self.enc_4 = self.encoder.blocks.group_4

        # fix the encoder
        for i in range(4):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [self.input_layer(image)]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class MSGDiscFeatureExtractor(nn.Module):
    def __init__(self, config, disc_model):
        super().__init__()
        # can't use list like other extractors... since the block is not sequential... Converting it into a list will loose that info
        # as it'll line it up sequentially.
        self.config = config
        self.depth= self.config.model_config.discriminator.depth
        self.rgb_to_features = disc_model.rgb_to_features
        self.layers = disc_model.layers

        # fix the encoder
        for i in self.layers:
            i.requires_grad = False

    def forward(self, inputs):
        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales, total no. of discriminator layers must match; inputs!"
        with torch.no_grad():
            y = self.rgb_to_features[self.depth - 2](inputs[self.depth - 1])
            y = self.layers[self.depth - 1](y)

        FeatMaps=[y]
        for x, block, converter in \
                zip(reversed(inputs[1:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = torch.cat((input_part, y), dim=1)  # concatenate the inputs:
            with torch.no_grad():
                y = block(y)  # apply the block
            FeatMaps.append(y)
            """
            torch.Size([8, 128, 56, 56])
            torch.Size([8, 256, 28, 28])
            torch.Size([8, 256, 14, 14])
            torch.Size([8, 256, 7, 7])
            """

        return FeatMaps

class DiscFeatureExtractor(nn.Module):
    def __init__(self, config, disc_model):
        super(DiscFeatureExtractor, self).__init__()
        self.config = config
        # loads the full dall_e model; encoder + decoder
        self.enc_1 = disc_model.conv1
        self.enc_2 = disc_model.conv2
        self.enc_3 = disc_model.conv3
        self.enc_4 = disc_model.conv4

        # fix the encoder
        for i in range(4):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    s = img.size()
    h = s[2]
    w = s[3]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(nn.functional.interpolate(img,
                            size=[nh, nw], mode='bilinear',
                            align_corners=True))
    return scaled_imgs

class GANLoss_Base():
    """
    Base class for all GAN losses
    """

    def __init__(self):
        pass 
        
    def disc_loss(self, real_samps, fake_samps):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("discriminator loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("generator loss method has not been implemented")
