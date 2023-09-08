import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config 
        self.model_config = self.config.model_config
        self.gan_loss_type= self.model_config.gan_loss_type

        ndf = 64
        nc = 3

        self.ndf = ndf
        self.nc = nc

        ################### DISCRIMINATOR MODEL ###################
        # input is (nc) x 64 x 64
        self.conv1 = nn.Sequential(nn.Conv2d(nc, ndf, 7, 2, 1, bias=False),
                                   nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 5, 2, 1, bias=False),
                                   nn.BatchNorm2d(ndf * 2),
                                   nn.LeakyReLU(0.2, inplace=True))
                                   # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 0, bias=False),
                                   nn.BatchNorm2d(ndf * 4),
                                   nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 0, bias=False),
                                   nn.BatchNorm2d(ndf * 8),
                                   nn.LeakyReLU(0.2, inplace=True))
        # conv layer;
        self.conv5 = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(ndf * 8),
                                   nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf*8) x 4 x 4
        self.conv6 = nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False)
        ################### DISCRIMINATOR MODEL ###################
        
        # last layer in wgan is not a sigmoid; 
        # wgan does not treat discriminator as a classifier; more like a critic
        # output is simply the logits; so that wasserstein distance can be calculated
        # for std gan don't need sigmoid, since BCEwithLogitLoss is used; which internally adds a sigmoid.
        self.final_layer = nn.Linear(4 * 4, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        feats = self.conv1(x) # torch.Size([8, 64, 110, 110])
        feats = self.conv2(feats) # torch.Size([8, 128, 54, 54])
        feats = self.conv3(feats) # torch.Size([8, 256, 25, 25])
        feats = self.conv4(feats) # torch.Size([8, 512, 12, 12])
        feats = self.conv5(feats) # torch.Size([8, 512, 6, 6])
        feats = self.conv6(feats) # torch.Size([8, 1, 4, 4])
        #if self.gan_loss_type!='std':
        feats = feats.view(feats.size(0), -1)
        feats = self.final_layer(feats)
        if self.gan_loss_type=='wgan':
            return feats 
        else:
            return self.sigmoid(feats).view(-1)

class MSGDiscriminator(nn.Module):
    """input images should be: 
    torch.Size([8, 3, 224, 224])
    torch.Size([8, 3, 112, 112])
    torch.Size([8, 3, 56, 56])
    torch.Size([8, 3, 28, 28])
    torch.Size([8, 3, 14, 14])
    torch.Size([8, 3, 7, 7])

    for a 6 scale pyramid , i.e. depth = 6
    """
    def __init__(self, config):
        super(MSGDiscriminator, self).__init__()
        self.config = config 
        self.model_config = self.config.model_config
        self.discriminator_config = self.model_config.discriminator
        self.gan_loss_type= self.model_config.gan_loss_type
        self.depth= self.discriminator_config.depth
        layer_type= self.discriminator_config.conv_layer_type
        if layer_type=='equal':
            use_equalized_conv = True
        else:
            use_equalized_conv = False 

        self.feature_size=self.discriminator_config.feature_size_ndf # ndf = 64
        self.input_channels=self.discriminator_config.input_channels_nc # nc = 6

        assert self.feature_size != 0 and ((self.feature_size & (self.feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if self.depth >= 4:
            assert self.feature_size >= np.power(2, self.depth - 4), \
                "feature size cannot be produced"

        # create the fromRGB layers for various inputs:
        self.rgb_to_features = nn.ModuleList()
        self.final_converter = self.from_rgb(self.feature_size // 2, use_equalized_conv)

        # create a module list of the other required general convolution blocks
        self.layers = nn.ModuleList()
        self.final_block = MSGGAN_DisFinalBlock(self.feature_size, use_eql=use_equalized_conv)

        ################### DISCRIMINATOR MODEL ###################
        # create the remaining layers
        for i in range(self.depth):
            if i > 2:
                layer = MSGGAN_DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    use_eql=use_equalized_conv
                )
                rgb = self.from_rgb(int(self.feature_size // np.power(2, i - 1)), use_equalized_conv)
                #print('layer in_dim:',int(self.feature_size // np.power(2, i - 2)), 'layer out_dim:',int(self.feature_size // np.power(2, i - 2)))
                #print('rgb out_dim:', int(self.feature_size // np.power(2, i - 1)))
            else:
                layer = MSGGAN_DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            use_eql=use_equalized_conv)
                rgb = self.from_rgb(self.feature_size // 2,use_equalized_conv)
                #print('layer in_dim:',self.feature_size, 'layer out_dim:',self.feature_size // 2)
                #print('rgb out_dim:',self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 2] = \
            self.from_rgb(self.feature_size // np.power(2, i - 2), use_equalized_conv)
        ################### DISCRIMINATOR MODEL ###################
        
        self.final_linear = nn.Linear(4 * 4, 1)
        # incase not using wgan;
        self.sigmoid = nn.Sigmoid()

    def from_rgb(self, out_channels, use_equalized_conv):
        # For converting input RGB images into arbitrary feature maps of dim: out_channels
        if use_equalized_conv:
            return _equalized_conv2d(3, out_channels, (1, 1), bias=True) 
        else:
            return nn.Conv2d(3, out_channels, (1, 1), bias=True)
    
    def to_rgb(self, in_channels, use_equalized_conv):
        # For converting feature maps into RGB images
        if use_equalized_conv:
            return _equalized_conv2d(in_channels, 3, (1, 1), bias=True) 
        else:
            return nn.Conv2d(in_channels, 3, (1, 1), bias=True)

    def cat(self, prev_feat, scaled_input):
        return torch.cat((prev_feat, scaled_input), dim=1)
        
    def forward(self, inputs):
        # Input is always an image tensor; i.e. batch x 3 x height x width, where 3 = channels index.
        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales, total no. of discriminator layers must match; inputs!"

        y = self.rgb_to_features[self.depth - 2](inputs[self.depth - 1])
        y = self.layers[self.depth - 1](y)

        for x, block, converter in \
                zip(reversed(inputs[1:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = torch.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block
            """
            torch.Size([8, 128, 56, 56])
            torch.Size([8, 256, 28, 28])
            torch.Size([8, 256, 14, 14])
            torch.Size([8, 256, 7, 7])
            """

        # calculate the final block:
        input_part = self.final_converter(inputs[0])
        y = torch.cat((input_part, y), dim=1)
        y = self.final_block(y) #torch.Size([8, 1, 4, 4]), after flattening > torch.Size([16])
        y = self.final_linear(y)

        if self.gan_loss_type=='wgan':
            return y 
        else:
            return self.sigmoid(y).view(-1)

############################ Helper Functions  ############################
# ==========================================================
# Equalized learning rate blocks:
# extending Conv2D and Deconv2D layers for equalized learning rate logic
# ==========================================================
class _equalized_conv2d(nn.Module):
    """ conv2d with the concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod

        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv2d

        return conv2d(input=x,
                      weight=self.weight * self.scale,  # scale the weight on runtime
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class _equalized_deconv2d(nn.Module):
    """ Transpose convolution using the equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out: output channels
            :param k_size: kernel size
            :param stride: stride for convolution transpose
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt

        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_in, c_out, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconv
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv_transpose2d

        return conv_transpose2d(input=x,
                                weight=self.weight * self.scale,  # scale the weight on runtime
                                bias=self.bias if self.use_bias else None,
                                stride=self.stride,
                                padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class _equalized_linear(nn.Module):
    """ Linear layer using equalized learning rate
        Args:
            :param c_in: number of input channels
            :param c_out: number of output channels
            :param bias: whether to use bias with the linear layer
    """

    def __init__(self, c_in, c_out, bias=True):
        """
        Linear layer modified for equalized learning rate
        """
        from numpy import sqrt

        super().__init__()

        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in)
        ))

        self.use_bias = bias

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import linear
        return linear(x, self.weight * self.scale,
                      self.bias if self.use_bias else None)


# -----------------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference:
# https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# -----------------------------------------------------------------------------------
class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y

# function to calculate the Exponential moving averages for the Generator weights
# This function updates the exponential average weights based on the current training
def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)

class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y

class MSGGAN_DisGeneralConvBlock(nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, use_eql=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import AvgPool2d, LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
        else:
            # convolutional modules
            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3),
                                padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3),
                                padding=1, bias=True)

        self.downSampler = AvgPool2d(2)  # downsampler

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y


class MSGGAN_DisFinalBlock(nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels, use_eql=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4),
                                            bias=True)

            # final layer emulates the fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)

        else:
            # modules required:
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)

            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation
        # flatten the output raw discriminator scores
        return y.view(y.size(0), -1)