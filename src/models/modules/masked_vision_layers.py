import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from src.models.modules.pos_embeds import *
from src.models.modules.layer_utils import Flatten
import timm
from timm.models.vision_transformer import PatchEmbed, Block

"""
layers for: https://arxiv.org/abs/2111.06377
inspiration: https://github.com/facebookresearch/mae 
"""

class MAEEncoder(nn.Module):
    def __init__(self, config, patch_embed, norm_layer):
        super(MAEEncoder, self).__init__()

        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config =  self.config.dataset_config
        # encoder args;
        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        num_heads = self.model_config.image_encoder.num_heads
        mlp_ratio = self.model_config.image_encoder.mlp_ratio
        depth = self.model_config.image_encoder.depth
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = patch_embed #PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()
    
    def forward(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    # ----------------- helper functions --------------------
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        #torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
      
class MSGMAEEncoder(nn.Module):
    def __init__(self, config, patch_embed, norm_layer):
        super(MSGMAEEncoder, self).__init__()

        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config =  self.config.dataset_config
        # encoder args;
        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        num_heads = self.model_config.image_encoder.num_heads
        mlp_ratio = self.model_config.image_encoder.mlp_ratio
        depth = self.model_config.image_encoder.depth
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = patch_embed #PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()
    
    def forward(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        prev_feat_maps=[]
        for i, blk in enumerate(self.blocks):
            x = blk(x) # all of size: [batch, 50, 768]
            # only need the encoder maps that correspond to the decoder depth layers
            if i >= (self.model_config.image_encoder.depth - self.model_config.image_decoder.decoder_depth) and i < self.model_config.image_encoder.depth-1:
                prev_feat_maps.append(x)
        x = self.norm(x)
        # don't include the final element; as we need it post normalisation.
        return x, mask, ids_restore, prev_feat_maps

    # ----------------- helper functions --------------------
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        #torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def static_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

class MAEDecoder(nn.Module):
    def __init__(self, config, patch_embed, norm_layer):
        super(MAEDecoder, self).__init__()

        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config = self.config.dataset_config

        # decoder args;
        embed_dim = self.model_config.image_encoder.embed_dim
        decoder_embed_dim = self.model_config.image_decoder.decoder_embed_dim
        decoder_num_heads = self.model_config.image_decoder.decoder_num_heads
        decoder_depth = self.model_config.image_decoder.decoder_depth
        mlp_ratio = self.model_config.image_encoder.mlp_ratio

        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        
        self.patch_embed = patch_embed #PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True) # decoder to patch

        self.initialize_weights()
    
    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks (8 decoder blocks)
        for blk in self.decoder_blocks:
            x = blk(x) # all size: [batch_size, 197, 512]
        x = self.decoder_norm(x) # size: [batch_size, 197, 512]

        # predictor projection
        x = self.decoder_pred(x) # size: [batch_size, 197, 768]

        # remove cls token
        x = x[:, 1:, :] # size: [batch_size, 196, 768]

        return x

    # ------------------ helper functions ------------------------
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MSGMAEDecoder(nn.Module):
    def __init__(self, config, patch_embed, norm_layer):
        super(MSGMAEDecoder, self).__init__()

        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config = self.config.dataset_config

        # decoder args;
        embed_dim = self.model_config.image_encoder.embed_dim
        decoder_embed_dim = self.model_config.image_decoder.decoder_embed_dim
        decoder_num_heads = self.model_config.image_decoder.decoder_num_heads
        decoder_depth = self.model_config.image_decoder.decoder_depth
        self.depth = decoder_depth
        mlp_ratio = self.model_config.image_encoder.mlp_ratio

        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        
        self.patch_embed = patch_embed #PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        # image scales:
        self.img_scales = self.get_pyramid_scale(img_size, self.model_config.discriminator.depth)
        
        # create a module list of the other required layers
        # u-net compression;
        self.compression_layers = nn.ModuleList([self.create_compression_layers(decoder_embed_dim, decoder_embed_dim)]) #nn.Linear(decoder_embed_dim*2, decoder_embed_dim)
        for i in range(decoder_depth-1):
            self.compression_layers.append(self.create_compression_layers(decoder_embed_dim, decoder_embed_dim))

        #self.skip_feature_layers = nn.ModuleList([nn.Linear(768, 512)]) #nn.Linear(decoder_embed_dim*2, decoder_embed_dim)
        #for i in range(decoder_depth-1):
        #    self.skip_feature_layers.append(nn.Linear(768, 512))
        
        # converting feature map to rgb image for MSG Discriminator:
        self.rgb_converters = nn.ModuleList([self.to_rgb(kernel_size=3, stride=2, padding=1, use_pool=False, pool_val=None)])
        for i in range(1,5):
            self.rgb_converters.append(self.to_rgb(kernel_size=5, stride=2, padding=2, use_pool=True, pool_val=2**i))

        # converting feature maps to patches:
        self.patch_converters = nn.ModuleList([self.to_patch(decoder_embed_dim, patch_size**2 * in_channels)])
        for i in range(5):
            self.patch_converters.append(self.to_patch(decoder_embed_dim, patch_size**2 * in_channels))

        # pad the lists where you don't want to use them;
        self.compression_layers= self.pad_module_list(self.compression_layers, None, False) # module_list, pad_val=None, pad_end=True
        #self.skip_feature_layers= self.pad_module_list(self.skip_feature_layers, None, False) # module_list, pad_val=None, pad_end=True
        self.rgb_converters= self.pad_module_list(self.rgb_converters, None, True)
        self.patch_converters= self.pad_module_list(self.patch_converters, None, True)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True) # decoder to patch

        self.initialize_weights()
    
    def create_compression_layers(self, in_dim, out_dim):
        return nn.Linear(in_dim*2, out_dim)
    
    def to_rgb(self, kernel_size, stride=1, padding=1, use_pool=False, pool_val=2):
        # For converting feature maps into RGB images
        #return nn.Conv2d(in_channels, 3, (1, 1), bias=True)
        if use_pool:
            return nn.Sequential(nn.AvgPool2d(pool_val),
                                 nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, bias=True))

        else:
            return nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, bias=True)

    def to_patch(self, in_dim, out_dim):
        # For converting feature maps into RGB images
        return nn.Linear(in_dim, out_dim, bias=True)
    
    def pad_module_list(self, module_list, pad_val=None, pad_end=True):
        diff = self.depth - len(module_list)
        if pad_end:
            start_pad = diff - 1 # number of indexes to pad at the start
        else:
            start_pad = diff
        # padding start:
        if start_pad >= 1:
            for i in range(start_pad):
                module_list.insert(0, pad_val)

        if pad_end:
            # padding end:
            module_list.append(pad_val)
        return module_list

    def convert_feat_to_img(self, feature_map, patch_converter, rgb_converter):
        # feature map > patch projection > remove cls token > cvt to img (unpatchify) > reshape (downsample / upsample accordingly)
        feat = patch_converter(feature_map)
        # remove cls token
        feat = feat[:, 1:, :] # size: [batch_size, 196, 768]
        # cvt to img (unpatchify)
        feat_img = self.unpatchify(feat) # 224 * 224 * 3
        # cvt to rgb:
        rgb_img = rgb_converter(feat_img)
        return rgb_img

    def repurpose_prev_feat(self, feat_map, ids_restore):
        x = self.decoder_embed(feat_map)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed
        return x

    def forward(self, x, ids_restore, prev_feat_maps):
        # append the latest
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        
        rgb_outputs=[]
        # apply Transformer blocks
        # for i, blk in enumerate(self.decoder_blocks):
        for i, (blk, compression, rgb_converter, patch_converter) in enumerate(zip(self.decoder_blocks, self.compression_layers, self.rgb_converters, self.patch_converters)):
            if i == 0:
                # just run through decoder transformer layers; (only first layer)
                x = blk(x)                
            else:
                prev_feat = prev_feat_maps[-i]
                prev_feat = self.repurpose_prev_feat(prev_feat, ids_restore) # iterative unmasking.
                # U-Net compression: cat > compress > fwd propagate
                x = compression(torch.cat((x, prev_feat), dim=2))
                # run combined weights through the decoder block;
                x = blk(x)
                # get rgb frames if converters are not None:
                if rgb_converter!=None and patch_converter!=None:
                    # feature_map, patch_converter, rgb_converter
                    rgb_feat_img = self.convert_feat_to_img(x, patch_converter, rgb_converter)
                    rgb_outputs.append(rgb_feat_img) # 5 images at 5 scales; append final image later;


        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        # final output (rgb) and previous feature maps (converted to rgb)
        # remember in the model to append the x to the rgb_outputs list (if you do it here; it'll clone it and grads won't back propagate)
        rgb_outputs.insert(0,self.unpatchify(x))
        return x, rgb_outputs

    # ------------------ helper functions ------------------------
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
    
    def get_pyramid_scale(self, img_size, num_scales):
        h,w = img_size
        scaled_sizes=[]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_sizes.append([nh,nw])
        return scaled_sizes

# -------------------- Helper function to create a ViT model via timm ------------------------
class VisionTransformer_EncoderOnly(nn.Module):
    def __init__(self, config, norm_layer, drop_rate=0.):
        super(VisionTransformer_EncoderOnly, self).__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config =  self.config.dataset_config
        # encoder args;
        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        num_heads = self.model_config.image_encoder.num_heads
        mlp_ratio = self.model_config.image_encoder.mlp_ratio
        depth = self.model_config.image_encoder.depth
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome