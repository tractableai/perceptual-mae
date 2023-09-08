from src.models.modules.layer_utils import * 
import torch 
import torch.nn as nn 
#from einops import rearrange


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.config = config
        self.model_config = self.config.model_config.image_encoder

        self.encoder_arch= self.model_config.name
        self.pretrained = self.model_config.pretrained

        # transformer architecture / convolution
        self.encoder = build_image_encoder(arch=self.encoder_arch, pretrained=self.pretrained)
        s = self.config.dataset_config.max_images
        self.input_transform = nn.Conv2d(s*3, 3, 1)

        self.input_processing = self.model_config.input_processing

        # if not loading pre-trained weights from imagenet; initialise with xavier
        if self.pretrained!=True:
            # if not loading pre-trained weights; use xavier initialiser
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        if self.input_processing=='single_image':
            return self.encoder(x)
        # x = [B, C, S, H, W]
        elif self.input_processing=='concat':
            # reshape tensor;
            b, c, s, h, w = x.shape
            x = x.reshape(b, s*c, h, w)
            x = self.input_transform(x)
            output = self.encoder(x)
            return output 

        elif self.input_processing=='individual':
            # iterate over all images in sample;
            outputs=[]
            for i in range(x.size(2)):
                out = self.encoder(x[:,:,i,:,:])
                outputs.append(out)
            # torch.Size([32, 3, 10, 224, 224])
            outputs = torch.stack(outputs).permute(1,0,2)
            #outputs = self.combiner(outputs.reshape(b, s*c, h, w))
            #print(outputs.size())
            return outputs
                
        else:
            raise Exception('the following type of input processing has not been implemented: {}'.format(self.input_processing))
        


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y